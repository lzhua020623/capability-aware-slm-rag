"""Verify the identical frozen payload for archive and Kaggle extracted input layouts."""
import hashlib
import json
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

from scripts.load_answer_repair_bundle import load_frozen_bundle


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.work = self.base / "work"
        self.work.mkdir()
        self.inputs = self.base / "input"
        self.root = self.inputs / "private_dataset" / "frozen_gpu"
        (self.root / "src").mkdir(parents=True)
        (self.root / "src/model.py").write_text("immutable payload\n")
        self.checksums = {"src/model.py": digest(self.root / "src/model.py")}
        (self.root / "bundle_checksums.json").write_text(json.dumps(self.checksums))
        self.checksum_sha = digest(self.root / "bundle_checksums.json")
        self.archive = self.base / "frozen_gpu.tar.gz"
        with tarfile.open(self.archive, "w:gz") as handle:
            handle.add(self.root, arcname="frozen_gpu")
        self.archive_sha = digest(self.archive)

    def load(self):
        return load_frozen_bundle(work=self.work, input_root=self.inputs, bundle_name="frozen_gpu",
                                  archive_sha256=self.archive_sha, checksums_sha256=self.checksum_sha)

    def test_auto_extracted_layout_is_verified_copied_and_resumable(self):
        (self.root / "unverified.py").write_text("must not enter import path")
        bundle, archive, report = self.load()
        self.assertEqual(report["transport"], "kaggle_auto_extracted")
        self.assertFalse(report["archive_hash_verified"])
        self.assertIsNone(archive)
        self.assertEqual((bundle / "src/model.py").read_bytes(), (self.root / "src/model.py").read_bytes())
        self.assertFalse((bundle / "unverified.py").exists())
        self.assertEqual(self.load()[0], bundle)

    def test_intact_archive_checks_both_archive_and_payload_hashes(self):
        self.archive.rename(self.inputs / "private_dataset/frozen_gpu.tar.gz")
        bundle, archive, report = self.load()
        self.assertEqual(report["transport"], "intact_archive")
        self.assertTrue(report["archive_hash_verified"])
        self.assertEqual(digest(archive), self.archive_sha)
        self.assertEqual(digest(bundle / "src/model.py"), self.checksums["src/model.py"])

    def test_modified_payload_or_checksums_cannot_be_promoted(self):
        (self.root / "src/model.py").write_text("tampered")
        with self.assertRaisesRegex(ValueError, "Frozen file hash mismatch"):
            self.load()
        (self.root / "bundle_checksums.json").write_text(json.dumps({"src/model.py": digest(self.root / "src/model.py")}))
        with self.assertRaisesRegex(ValueError, "checksums document hash mismatch"):
            self.load()

    def test_corrupt_archive_does_not_fall_back_to_other_input(self):
        (self.work / "frozen_gpu.tar.gz").write_bytes(b"bad archive")
        with self.assertRaisesRegex(ValueError, "Archive hash mismatch"):
            self.load()

    def test_duplicate_input_and_existing_writable_mutation_are_rejected(self):
        bundle, _, _ = self.load()
        (bundle / "src/model.py").write_text("tampered working copy")
        with self.assertRaisesRegex(ValueError, "Existing writable bundle differs"):
            self.load()
        other = self.inputs / "another" / "frozen_gpu"
        other.mkdir(parents=True)
        (other / "bundle_checksums.json").write_bytes((self.root / "bundle_checksums.json").read_bytes())
        with self.assertRaisesRegex(ValueError, "found 2 extracted roots"):
            self.load()

    def test_missing_input_fails_before_model_or_pipeline_calls(self):
        (self.root / "bundle_checksums.json").unlink()
        with self.assertRaisesRegex(ValueError, "found 0 extracted roots"):
            self.load()

    def test_real_frozen_payload_all_arms_can_be_cpu_replayed(self):
        # Test-only model stubs; no predictions produced here are research results.
        repo = Path(__file__).resolve().parents[1]
        archive = repo / "artifacts/answer_repair/v8_nq75/answer_repair_v8_nq75_gpu.tar.gz"
        with tarfile.open(archive) as handle:
            handle.extractall(self.base / "frozen", filter="data")
        bundle = self.base / "frozen/answer_repair_v8_nq75_gpu"
        results = self.base / "simulated_results"
        driver = r'''
import contextlib, io, json, sys
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, sys.argv[1])
from scripts.run_answer_repair import main
from src.compensation.answer_repair import Generation
bundle, results = Path(sys.argv[1]), Path(sys.argv[2])
class TestGenerator:
    runtime_environment = {"fixture": "CPU stub; not inference"}
    def __init__(self, output): self.output = output
    def generate(self, prompt, max_new_tokens): return Generation(self.output, .01, 10, 5)
uncertain = "\n".join(k + ": uncertain" for k in ("ENTITY", "RELATION", "TYPE", "COVERAGE", "SUPPORT", "CONFLICT_FREE")) + "\nRANK: 0\nQUOTE: NONE\nREASON: Test fixture only, no model inference."
for arm in json.loads((bundle / "manifest.json").read_text())["protocol"]:
    strategy = arm["strategy"]
    output = "INSUFFICIENT" if strategy in ("short_answer_quote", "conservative_no_original_gate") else uncertain
    with contextlib.redirect_stdout(io.StringIO()), patch("src.compensation.qwen_repair.QwenRepairGenerator", return_value=TestGenerator(output)):
        main(["--dataset", "nq", "--c1", str(bundle / "inputs/nq75_c1.jsonl"), "--all", "--strategy", strategy, "--out", str(results / (strategy + ".jsonl"))])
for name in ("manifest.json", "bundle_checksums.json"):
    (results / name).write_bytes((bundle / name).read_bytes())
'''
        run = subprocess.run([sys.executable, "-c", driver, str(bundle), str(results)],
                             capture_output=True, text=True, timeout=60)
        self.assertEqual(run.returncode, 0, run.stderr)
        audit = self.base / "cpu_audit"
        cmd = [sys.executable, str(repo / "scripts/audit_answer_repair_release.py"),
               "--bundle-dir", str(bundle), "--results-dir", str(results), "--out-dir", str(audit)]
        replay = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        self.assertEqual(replay.returncode, 0, replay.stderr)
        report = json.loads((audit / "cpu_replay.json").read_text())
        self.assertEqual(report["replayed_records"], 300)
        self.assertEqual(report["unique_dataset_sample_ids"], 75)
        self.assertEqual(report["manual_review_status"], "pending_output_review")
        self.assertFalse(report["semantic_correctness_verified"])
        # A forged final status must not survive deterministic trace replay.
        path = results / "conservative_repair.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()]
        records[0]["status"] = "original_supported"
        path.write_text("".join(json.dumps(row) + "\n" for row in records))
        rejected = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("mismatch", rejected.stderr)


if __name__ == "__main__":
    unittest.main()
