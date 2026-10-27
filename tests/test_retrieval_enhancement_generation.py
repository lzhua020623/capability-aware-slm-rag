"""Fake generation, durable resume, and fully separate offline answer metrics."""

from copy import deepcopy
import fcntl
import json
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts.evaluate_nq_p320_generation import accepted_metadata, evaluate_generation
from scripts.run_nq_p320_gpu_generation import frozen_metadata, verify_bundle
from scripts.prepare_nq_p320_generation import runtime_code_paths
from src.compensation.retrieval_enhancement.generation_replay import (
    load_generation_inputs, read_results, run_generation, sha256_file,
)


def source(sid, initial="wrong"):
    return {"sample_id": str(sid), "question": f"Question {sid}?", "initial_answer": initial,
            "evidence": [{"rank": i, "passage_id": f"{sid}_p{i}", "text": "Ada lived in London."}
                         for i in range(1, 6)]}


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.input = self.root / "input.jsonl"
        self.output = self.root / "gpu_results"
        self.rows = [source("9007199254740993123"), source("b"), source("c")]
        self.write()
        self.calls = []

    def write(self):
        self.input.write_text("".join(json.dumps(r) + "\n" for r in self.rows))

    def generate(self, prompt):
        self.calls.append(prompt)
        return "London"

    def run_worker(self, **kwargs):
        return run_generation(self.input, self.output, self.generate, run_metadata={"frozen": True}, **kwargs)

    def test_success_and_ordered_resume_call_only_pending_samples(self):
        first = self.run_worker(limit=1)
        self.assertEqual((first["completed"], first["remaining"]), (1, 2))
        before = (self.output / "results.jsonl").read_bytes()
        second = self.run_worker(resume=True)
        self.assertEqual((second["resumed"], second["new_completed"]), (1, 2))
        self.assertTrue((self.output / "results.jsonl").read_bytes().startswith(before))
        records = read_results(self.output / "results.jsonl")
        self.assertEqual([r["sample_id"] for r in records], [r["sample_id"] for r in self.rows])
        self.assertTrue(all(r["status"] == "answer_replaced" for r in records))
        self.assertEqual(len(self.calls), 3)
        final = self.run_worker(resume=True)
        self.assertEqual(final["new_completed"], 0)
        self.assertEqual(len(self.calls), 3)

    def test_refusal_and_ungrounded_raw_answers_are_saved_with_fallback(self):
        answers = iter(["", "UNKNOWN", "Paris"])
        run_generation(self.input, self.output, lambda _: next(answers), run_metadata={})
        records = read_results(self.output / "results.jsonl")
        self.assertEqual([r["raw_answer"] for r in records], ["", "UNKNOWN", "Paris"])
        self.assertTrue(all(r["status"] == "kept_original" and r["final_answer"] == "wrong" for r in records))
        self.assertTrue(all(r["latency"]["total"] >= r["latency"]["generation"] for r in records))

    def test_exception_is_saved_then_stops_without_excluding_sample(self):
        def fail(_):
            raise RuntimeError("fake CUDA OOM")
        summary = run_generation(self.input, self.output, fail, run_metadata={"frozen": True})
        self.assertTrue(summary["stopped_on_error"])
        self.assertEqual(summary["completed"], 1)
        record = read_results(self.output / "results.jsonl")[0]
        self.assertEqual(record["status"], "generation_failed")
        self.assertEqual(record["final_answer"], "wrong")
        resumed = self.run_worker(resume=True)
        self.assertEqual(resumed["resumed"], 1)
        self.assertEqual(len(self.calls), 2)

    def test_process_interrupt_keeps_completed_results(self):
        n = 0
        def interrupt(_):
            nonlocal n
            n += 1
            if n == 2:
                raise KeyboardInterrupt()
            return "London"
        with self.assertRaises(KeyboardInterrupt):
            run_generation(self.input, self.output, interrupt, run_metadata={"frozen": True})
        self.assertEqual(len(read_results(self.output / "results.jsonl")), 1)
        self.assertEqual(self.run_worker(resume=True)["resumed"], 1)

    def test_partial_tail_preserved_as_backup_and_repaired_only_by_resume(self):
        self.run_worker(limit=1)
        path = self.output / "results.jsonl"
        with path.open("ab") as f:
            f.write(b'{"sample_id":')
        with self.assertRaises(ValueError):
            read_results(path)
        summary = self.run_worker(resume=True)
        self.assertEqual(summary["completed"], 3)
        backups = list(self.output.glob("*.partial.*.bin"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), b'{"sample_id":')

    def test_complete_corruption_is_not_silently_repaired(self):
        self.run_worker(limit=1)
        path = self.output / "results.jsonl"
        path.write_bytes(path.read_bytes().replace(b'"London"', b'"Paris"'))
        with self.assertRaisesRegex(ValueError, "corrupt"):
            self.run_worker(resume=True)

    def test_input_or_config_change_blocks_resume_before_model_call(self):
        self.run_worker(limit=1)
        self.rows[1]["question"] = "Changed?"
        self.write()
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            self.run_worker(resume=True)
        self.assertEqual(len(self.calls), 1)
        with self.assertRaises(ValueError):
            run_generation(self.input, self.output, self.generate, run_metadata={"frozen": False}, resume=True)

    def test_existing_output_requires_explicit_resume_and_lock(self):
        self.run_worker(limit=1)
        with self.assertRaises(FileExistsError):
            self.run_worker()
        with (self.output / "writer.lock").open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                self.run_worker(resume=True)

    def test_gold_labels_and_extra_evidence_fields_are_rejected(self):
        original = deepcopy(self.rows)
        for key in ["gold_answer", "gold_passage_id", "short_answer", "label", "target_label", "em"]:
            self.rows = deepcopy(original); self.rows[0][key] = "forbidden"; self.write()
            with self.assertRaises(ValueError):
                load_generation_inputs(self.input)
        self.rows = deepcopy(original); self.rows[0]["evidence"][0]["gold"] = True; self.write()
        with self.assertRaises(ValueError):
            load_generation_inputs(self.input)

    def test_invalid_ids_ranks_and_duplicate_records_fail_preflight(self):
        for mutate in [lambda r: r[0].update(sample_id=1),
                       lambda r: r[0]["evidence"][0].update(rank=True),
                       lambda r: r[0]["evidence"].pop(),
                       lambda r: r.append(deepcopy(r[0]))]:
            self.rows = [source("a"), source("b")]; mutate(self.rows); self.write()
            with self.assertRaises(ValueError):
                self.run_worker()
            self.assertFalse(self.calls)
            self.assertFalse(self.output.exists())

    def test_gpu_cli_import_and_fake_worker_require_no_model_packages(self):
        code = '''
import sys, tempfile, json
from pathlib import Path
for name in ['torch', 'transformers', 'bitsandbytes', 'faiss', 'sentence_transformers']:
    sys.modules[name] = None
from scripts.run_nq_p320_gpu_generation import lazy_qwen_generator
from tests.test_retrieval_enhancement_generation import source
from src.compensation.retrieval_enhancement.generation_replay import run_generation
with tempfile.TemporaryDirectory() as t:
    p = Path(t)/'input.jsonl'; p.write_text(json.dumps(source('x'))+'\\n')
    r = run_generation(p, Path(t)/'out', lambda prompt: 'London', run_metadata={})
    assert r['completed'] == 1
'''
        r = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True,
                           cwd=Path(__file__).resolve().parents[1])
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_bundle_checksums_fail_before_model_calls(self):
        manifest = {"files": {"input.jsonl": sha256_file(self.input),
                              "generation_inputs.jsonl": sha256_file(self.input)}}
        (self.root / "generation_inputs.jsonl").write_bytes(self.input.read_bytes())
        (self.root / "package_manifest.json").write_text(json.dumps(manifest))
        with patch('scripts.run_nq_p320_gpu_generation.ROOT', self.root):
            verify_bundle(self.input)
            self.input.write_text("tampered")
            with self.assertRaisesRegex(ValueError, "checksum"):
                verify_bundle(self.input)
        self.assertFalse(self.calls)

    def test_standalone_runtime_allowlist_contains_all_import_dependencies(self):
        repo = Path(__file__).resolve().parents[1]
        bundle = self.root / "isolated_bundle"
        for name in runtime_code_paths():
            dest = bundle / name; dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(repo / name, dest)
        code = '''
import sys
for name in ['torch','transformers','bitsandbytes','faiss','sentence_transformers']:
    sys.modules[name]=None
from scripts.run_nq_p320_gpu_generation import frozen_metadata
from src.compensation.retrieval_enhancement.generation_replay import load_generation_inputs
from src.generation.prompts import build_rag_prompt
assert frozen_metadata()['generator']['max_new_tokens']==64
assert 'Question: Fiction?' in build_rag_prompt('Fiction?', [{'text': 'Fictional evidence.'}])
'''
        r = subprocess.run([sys.executable, "-B", "-c", code], cwd=bundle, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)


class OfflineMetricsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.input = self.root / "input.jsonl"; self.gpu = self.root / "gpu"
        self.refs = self.root / "offline_gold.jsonl"; self.report = self.root / "metrics"
        self.inputs = [source("a"), source("b", "Paris"), source("c", "Paris"), source("d", "London")]
        self.input.write_text("".join(json.dumps(r) + "\n" for r in self.inputs))
        self.references = [
            {"sample_id": "a", "initial_answer": "wrong", "label": "evidence_insufficiency", "gold_answers": ["London"], "generation_required": True},
            {"sample_id": "b", "initial_answer": "Paris", "label": "correct", "gold_answers": ["Paris"], "generation_required": True},
            {"sample_id": "c", "initial_answer": "Paris", "label": "correct", "gold_answers": ["Paris"], "generation_required": True},
            {"sample_id": "d", "initial_answer": "London", "label": "evidence_accessibility", "gold_answers": ["London"], "generation_required": True},
            {"sample_id": "e", "initial_answer": "wrong", "label": "evidence_utilisation", "gold_answers": ["right"], "generation_required": False},
            {"sample_id": "f", "initial_answer": "wrong", "label": "correct", "gold_answers": ["right"], "generation_required": False},
        ]
        self.refs.write_text("".join(json.dumps(r) + "\n" for r in self.references))

    def generate(self, *, metadata=None, **kwargs):
        answers = iter(["London", "Paris", "London", "UNKNOWN"])
        return run_generation(self.input, self.gpu, lambda _: next(answers),
                              run_metadata=frozen_metadata() if metadata is None else metadata, **kwargs)

    def evaluate(self):
        return evaluate_generation(self.refs, self.input, self.gpu, self.report, expected_count=6)

    def test_explicit_win2_acceptance_still_rejects_source_and_config_changes(self):
        package = Path(__file__).resolve().parents[1] / "docs/provenance/nq_p320_win2_package_manifest.json"
        # A fictional Windows run carries the approved package's source hashes,
        # not the newer Controller checkout's FEVER/prompt compatibility code.
        expected, _ = accepted_metadata(package)
        self.generate(metadata=expected)
        result = evaluate_generation(self.refs, self.input, self.gpu, self.report,
                                     expected_count=6, package_manifest=package)
        self.assertEqual(result["source_verification"]["mode"], "explicit_win2_manifest")
        manifest_path = self.gpu / "run_manifest.json"
        original = json.loads(manifest_path.read_text())
        for field in ("source_sha256", "generator"):
            bad = deepcopy(original)
            if field == "source_sha256":
                bad["run_metadata"][field]["src/generation/qwen.py"] = "0" * 64
            else:
                bad["run_metadata"][field]["max_new_tokens"] = 65
            manifest_path.write_text(json.dumps(bad))
            target = self.root / ("rejected_" + field)
            with self.assertRaisesRegex(ValueError, "hashes differ"):
                evaluate_generation(self.refs, self.input, self.gpu, target,
                                    expected_count=6, package_manifest=package)
            self.assertFalse(target.exists())

    def test_win2_manifest_does_not_accept_new_checkout_sources(self):
        self.generate()
        package = Path(__file__).resolve().parents[1] / "docs/provenance/nq_p320_win2_package_manifest.json"
        with self.assertRaisesRegex(ValueError, "hashes differ"):
            evaluate_generation(self.refs, self.input, self.gpu, self.report,
                                expected_count=6, package_manifest=package)
        self.assertFalse(self.report.exists())

    def test_before_after_harm_rejection_and_false_rejection_denominators(self):
        self.generate()
        r = self.evaluate()
        self.assertEqual(r["n"], 6)
        self.assertEqual(r["overall"]["base_em"], 0.5)
        self.assertEqual(r["overall"]["final_em"], 0.5)
        self.assertEqual(r["overall"]["em_delta"], 0)
        self.assertEqual(r["overall"]["guard_rejection_rate"], 0.5)
        self.assertEqual(r["overall"]["rejected_raw_em_correct_rate"], 0.5)
        correct = r["by_label"]["correct"]
        self.assertEqual(correct["harm_rate_among_baseline_em_correct"], 0.5)
        self.assertAlmostEqual(correct["em_harm_over_all_label_samples"], 1/3)
        self.assertEqual(r["by_label"]["evidence_utilisation"]["guard_rejection_rate"], None)
        self.assertEqual(r["overall"]["unchanged"], 2)
        with self.assertRaises(FileExistsError):
            self.evaluate()

    def test_partial_gpu_results_are_not_scored_as_complete(self):
        self.generate(limit=1)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.evaluate()
        self.assertFalse(self.report.exists())

    def test_partial_answer_f1_and_missing_unchanged_reference(self):
        self.references[0]["gold_answers"] = ["London UK"]
        self.refs.write_text("".join(json.dumps(r) + "\n" for r in self.references))
        self.generate()
        r = self.evaluate()
        self.assertEqual(r["by_label"]["evidence_insufficiency"]["final_em"], 0)
        self.assertAlmostEqual(r["by_label"]["evidence_insufficiency"]["f1_delta"], 2/3)
        self.refs.write_text("".join(json.dumps(r) + "\n" for r in self.references[:-1]))
        with self.assertRaisesRegex(ValueError, "count/uniqueness"):
            evaluate_generation(self.refs, self.input, self.gpu, self.root / "second_report", expected_count=6)

    def test_failures_remain_in_full_dataset_denominator(self):
        def fail(_):
            raise RuntimeError("fake failure")
        run_generation(self.input, self.gpu, fail, run_metadata=frozen_metadata())
        # Resume keeps the explicit failed row and completes the other three.
        run_generation(self.input, self.gpu, lambda _: "London", run_metadata=frozen_metadata(), resume=True)
        r = self.evaluate()
        self.assertEqual(r["overall"]["generation_errors"], 1)
        self.assertEqual(r["n"], 6)

    def test_changed_membership_or_baseline_cannot_be_silently_joined(self):
        self.generate()
        self.references[0]["initial_answer"] = "mismatching prediction"
        self.refs.write_text("".join(json.dumps(r) + "\n" for r in self.references))
        with self.assertRaisesRegex(ValueError, "initial answer"):
            self.evaluate()


if __name__ == "__main__":
    unittest.main()
