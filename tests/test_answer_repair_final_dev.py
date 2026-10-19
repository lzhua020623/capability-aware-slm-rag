"""Final development packaging cannot switch inference code or select by its results."""
import contextlib
import hashlib
import io
import json
import shutil
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import prepare_answer_repair_final_dev as final

ROOT = Path(__file__).parents[1]


class FinalDevelopmentTests(unittest.TestCase):
    def prepare(self, folder, revision, *, corrupt=False):
        root = Path(folder) / "project"
        for relative in (
            "artifacts/answer_repair/v8_nq75/answer_repair_v8_nq75_gpu.tar.gz",
            f"artifacts/answer_repair/v8{revision}_smoke8/answer_repair_v8{revision}_smoke8_gpu.tar.gz",
        ):
            dest = root / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, dest)
        baseline = root / "artifacts/answer_repair/v8r3_smoke8/result_validation.json"
        baseline.parent.mkdir(parents=True, exist_ok=True)
        baseline.write_text(json.dumps({"engineering_gate_passed": True, "test_fixture_only": True}))
        smoke = Path(folder) / "synthetic_selection_only"
        raw = smoke / "kaggle_20261019"
        raw.mkdir(parents=True)
        report = json.dumps({"selected_final_revision": revision, "ready_for_full_nq75_run": revision == "r4"})
        (raw / "smoke_acceptance.json").write_text(report)
        (smoke / "result_validation.json").write_text(json.dumps({
            "replayed_records": 32, "frozen_engineering_gate_recalculation_matches": True,
            "raw_file_sha256": {"smoke_acceptance.json": "bad" if corrupt else hashlib.sha256(report.encode()).hexdigest()},
            "test_fixture_only": True}))
        return root, smoke

    def run_build(self, folder, revision, *, corrupt=False):
        root, smoke = self.prepare(folder, revision, corrupt=corrupt)
        out, book = Path(folder) / "out", Path(folder) / "final.ipynb"
        with patch.object(final, "ROOT", root), contextlib.redirect_stdout(io.StringIO()):
            final.main(["--smoke-dir", str(smoke), "--out-dir", str(out), "--notebook", str(book)])
        return out, book

    def check_revision(self, revision):
        with tempfile.TemporaryDirectory() as folder:
            out, book = self.run_build(folder, revision)
            manifest = json.loads((out / "manifest.json").read_text())
            self.assertEqual(manifest["samples"], 75)
            self.assertEqual(manifest["groups"], {"include_e_core": 11, "correct_control": 64})
            self.assertEqual(manifest["selected_final_revision"], revision)
            self.assertEqual((manifest["expected_predictions"], manifest["max_generation_calls"]), (300, 600))
            self.assertFalse(manifest["tuning_allowed_after_run"])
            self.assertNotIn("semantic_regressions", manifest)
            self.assertNotIn("smoke_acceptance", manifest)
            archive = next(out.glob("*.tar.gz"))
            prefix = archive.name.removesuffix(".tar.gz") + "/"
            with tarfile.open(archive) as handle:
                for name, digest in manifest["runtime_source_sha256"].items():
                    self.assertEqual(hashlib.sha256(handle.extractfile(prefix + name).read()).hexdigest(), digest)
                human = handle.extractfile(prefix + "offline/initial_reviews.jsonl").read()
                self.assertEqual(hashlib.sha256(human).hexdigest(), manifest["initial_human_review_sha256"])
            code = "\n".join("".join(c["source"]) for c in json.loads(book.read_text())["cells"] if c["cell_type"] == "code")
            self.assertIn("manifest['samples'] == 75", code)
            self.assertIn("scripts/summarize_answer_repair_v8.py", code)
            self.assertNotIn("scripts/check_answer_repair_v8_smoke.py", code)
            self.assertNotIn("report['engineering_gate_passed']", code)
            for c in json.loads(book.read_text())["cells"]:
                if c["cell_type"] == "code":
                    compile("".join(c["source"]), "<final75-test>", "exec")

    def test_fallback_packages_r3_source_without_fact_contract(self):
        self.check_revision("r3")

    def test_pass_packages_r4_source_without_changing_runtime_code(self):
        self.check_revision("r4")

    def test_tampered_selection_receipt_refuses_a_package(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "verified frozen CPU replay"):
                self.run_build(folder, "r4", corrupt=True)
            self.assertFalse((Path(folder) / "out").exists())

    def test_no_gpu_results_cannot_be_treated_as_a_final_selection(self):
        with tempfile.TemporaryDirectory() as folder, self.assertRaises(FileNotFoundError):
            final.main(["--smoke-dir", folder, "--out-dir", str(Path(folder) / "not_created")])


if __name__ == "__main__":
    unittest.main()
