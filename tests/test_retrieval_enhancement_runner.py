"""Exercise the real Runner/metrics boundary using tiny files and injected Backend."""

import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import run_nq_retrieval_enhancement as runner
from src.compensation.retrieval_enhancement import EnhancementBudget


def hits(ids):
    return [{"passage_id": pid, "text": f"Evidence {pid}", "score": 0.8} for pid in ids]


class Backend:
    def __init__(self, *, fail_extra=False, fail_base=None):
        self.calls = []
        self.fail_extra = fail_extra
        self.fail_base = fail_base

    def retrieve(self, query, *, dataset, top_k):
        self.calls.append((query, dataset, top_k))
        if query == self.fail_base or (self.fail_extra and query == "Ada Lovelace birthplace"):
            raise RuntimeError("synthetic retrieval failure")
        return hits("FA" if query == "Ada Lovelace birthplace" else "ABCDE")


class Owner:
    def __init__(self):
        self.unloads = 0

    def unload(self):
        self.unloads += 1


class NQRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        # Preflight's presence checks are separate from these pure-Python tests.
        # Fake CLI runs must work even on machines without model/FAISS packages.
        dependencies = patch.object(runner.importlib.util, "find_spec", return_value=object())
        dependencies.start()
        self.addCleanup(dependencies.stop)
        self.root = Path(self.temp.name)
        self.queries = [{"id": 1, "question": "Where was Ada Lovelace born?", "gold_passage_id": "F", "short_answer": "London"},
                        {"id": 2, "question": "A query with constraints", "gold_passage_id": "A"}]
        self.manifest = {"dataset": "natural_questions", "seed": 42, "n": 2, "ids": ["1", "2"]}
        self.query_path = self.root / "queries.jsonl"
        self.manifest_path = self.root / "manifest.json"
        self.meta_path = self.root / "metadata.jsonl"
        self.index_path = self.root / "index"
        self.query_path.write_text("".join(json.dumps(x) + "\n" for x in self.queries))
        self.manifest_path.write_text(json.dumps(self.manifest))
        self.meta_path.write_text("".join(json.dumps(x) + "\n" for x in hits("ABCDEF")))
        self.index_path.write_bytes(b"NOT A REAL FAISS INDEX")
        self.cache = self.root / "hub"
        model = self.cache / "models--BAAI--bge-base-en-v1.5"
        (model / "refs").mkdir(parents=True)
        (model / "refs/main").write_text("fake-revision")
        self.snapshot = model / "snapshots/fake-revision"
        (self.snapshot / "1_Pooling").mkdir(parents=True)
        for name in ("config.json", "tokenizer_config.json", "1_Pooling/config.json"):
            (self.snapshot / name).write_text("{}")
        (self.snapshot / "modules.json").write_text(json.dumps([{"path": "1_Pooling"}]))
        (self.snapshot / "vocab.txt").write_text("tiny fake tokenizer")
        (self.snapshot / "model.safetensors").write_bytes(b"NOT REAL WEIGHTS")
        self.arguments = ["--queries", str(self.query_path), "--manifest", str(self.manifest_path),
                          "--metadata", str(self.meta_path), "--index", str(self.index_path),
                          "--cache-dir", str(self.cache)]

    def main(self, extra=()):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return runner.main([*self.arguments, *extra])

    def test_selection_preserves_manifest_order_and_strips_annotations(self):
        class Query(dict):
            def __getitem__(self, key):
                if key not in ("id", "question"):
                    raise AssertionError(f"annotation accessed: {key}")
                return super().__getitem__(key)
        manifest = dict(self.manifest, ids=["2", "1"])
        result = runner.select_queries([Query(x) for x in self.queries], manifest, limit=1)
        self.assertEqual(result, [{"sample_id": "2", "question": self.queries[1]["question"]}])

    def test_invalid_or_incomplete_sampling_fails_without_resampling(self):
        cases = [(self.queries * 2, self.manifest, None),
                 (self.queries[:1], self.manifest, None),
                 (self.queries, dict(self.manifest, ids=["1", "1"]), None),
                 (self.queries, dict(self.manifest, n=500), None),
                 (self.queries, dict(self.manifest, dataset="fever"), None),
                 (self.queries, self.manifest, 0), (self.queries, self.manifest, 3)]
        for records, manifest, limit in cases:
            with self.subTest(manifest=manifest, limit=limit), self.assertRaises(ValueError):
                runner.select_queries(records, manifest, limit)

    def test_runtime_selected_recovery_has_no_annotation_input(self):
        backend = Backend()
        result = runner.run_query("1", self.queries[0]["question"], backend, EnhancementBudget(max_retrieval_calls=1))
        self.assertEqual([x["passage_id"] for x in result["base_evidence"]], list("ABCDE"))
        self.assertEqual([x["passage_id"] for x in result["enhanced_evidence"]], list("AFBCD"))
        self.assertEqual(backend.calls, [(self.queries[0]["question"], "nq", 5), ("Ada Lovelace birthplace", "nq", 5)])
        self.assertFalse(any("gold" in key or "answer" in key for key in result))

    def test_zero_budget_is_paired_base_only_control(self):
        backend = Backend()
        result = runner.run_query("1", self.queries[0]["question"], backend, EnhancementBudget(max_retrieval_calls=0))
        self.assertEqual(result["base_evidence"], result["enhanced_evidence"])
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(result["trace"]["retrieval_calls"], 0)

    def test_additional_failure_preserves_base_and_counts_attempt(self):
        backend = Backend(fail_extra=True)
        result = runner.run_query("1", self.queries[0]["question"], backend, EnhancementBudget())
        self.assertEqual(result["base_evidence"], result["enhanced_evidence"])
        self.assertEqual(result["trace"]["status"], "retrieval_failed")
        self.assertEqual(result["trace"]["retrieval_calls"], 1)

    def test_default_preflight_checks_structure_without_loading_fake_weights_or_index(self):
        with patch.object(runner, "load_backend", side_effect=AssertionError("must not load")):
            report = runner.preflight(runner.parser().parse_args(self.arguments))
            self.assertTrue(report["structure_ready"], report["errors"])
            self.assertEqual(report["metadata_count"], 6)
            self.assertEqual(report["selected_count"], 2)
            self.assertFalse(report["index_loaded"])
            self.assertFalse(report["model"]["weights_loaded"])
            self.assertEqual(self.main(), 0)

    def test_missing_resources_block_run_before_loading_or_writing(self):
        self.index_path.unlink()
        output = self.root / "missing"
        with patch.object(runner, "load_backend") as load:
            self.assertEqual(self.main(["--run", "--output-dir", str(output)]), 2)
            load.assert_not_called()
        self.assertFalse(output.exists())

    def test_missing_python_dependencies_are_reported_without_importing(self):
        with patch.object(runner.importlib.util, "find_spec", return_value=None):
            report = runner.preflight(runner.parser().parse_args(self.arguments))
        self.assertFalse(report["structure_ready"])
        self.assertIn("missing Python dependency: faiss", report["errors"])

    def test_missing_shards_and_metadata_inconsistency_are_reported(self):
        (self.snapshot / "model.safetensors").unlink()
        (self.snapshot / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"x": "missing.safetensors"}}))
        self.assertIn("missing BGE weight shards", runner.check_model_cache(self.cache)["errors"])
        self.meta_path.write_text(json.dumps(hits("A")[0]) + "\n" + json.dumps(hits("A")[0]) + "\n")
        report = runner.preflight(runner.parser().parse_args(self.arguments))
        self.assertFalse(report["structure_ready"])
        self.assertTrue(any("duplicate IDs" in error for error in report["errors"]))

    def test_official_bge_stateless_normalize_needs_no_module_config(self):
        modules = json.loads((self.snapshot / "modules.json").read_text())
        modules.append({"idx": 2, "name": "2", "path": "2_Normalize",
                        "type": "sentence_transformers.models.Normalize"})
        (self.snapshot / "modules.json").write_text(json.dumps(modules))
        self.assertEqual(runner.check_model_cache(self.cache)["errors"], [])

    def test_stateful_pooling_still_requires_module_config(self):
        (self.snapshot / "1_Pooling/config.json").unlink()
        (self.snapshot / "modules.json").write_text(json.dumps([
            {"path": "1_Pooling", "type": "sentence_transformers.models.Pooling"}]))
        self.assertIn("missing BGE module config: 1_Pooling",
                      runner.check_model_cache(self.cache)["errors"])

    def test_runner_writes_traces_then_offline_labels_and_releases_resources(self):
        output = self.root / "run"
        owner, backend = Owner(), Backend()
        with patch.object(runner, "load_backend", return_value=(owner, backend)):
            self.assertEqual(self.main(["--run", "--limit", "1", "--output-dir", str(output)]), 0)
        self.assertEqual(owner.unloads, 1)
        traces = runner.load_jsonl(output / "traces.jsonl")
        self.assertEqual(len(traces), 1)
        self.assertNotIn("gold_passage_id", traces[0])
        self.assertNotIn("short_answer", traces[0])
        metrics = json.loads((output / "metrics.json").read_text())
        self.assertEqual(metrics["strict_id"]["base_recall_at5"], 0.0)
        self.assertEqual(metrics["strict_id"]["enhanced_recall_at5"], 1.0)
        config = json.loads((output / "run_config.json").read_text())
        self.assertEqual(config["scope"], "smoke_subset")
        self.assertEqual(config["sample_ids"], ["1"])
        self.assertEqual(config["inputs"]["queries"]["sha256"], runner.fingerprint(self.query_path)["sha256"])
        self.assertTrue(json.loads((output / "completion.json").read_text())["complete"])

    def test_base_failure_aborts_without_false_completion_and_releases_resources(self):
        output = self.root / "partial"
        owner, backend = Owner(), Backend(fail_base=self.queries[1]["question"])
        with patch.object(runner, "load_backend", return_value=(owner, backend)):
            self.assertEqual(self.main(["--run", "--output-dir", str(output)]), 2)
        self.assertEqual(owner.unloads, 1)
        self.assertEqual(len(runner.load_jsonl(output / "traces.jsonl")), 1)
        self.assertFalse((output / "completion.json").exists())
        self.assertFalse((output / "metrics.json").exists())

    def test_evaluate_only_does_not_require_model_index_or_retrieval(self):
        trace = runner.run_query("1", self.queries[0]["question"], Backend(), EnhancementBudget())
        path = self.root / "saved.jsonl"
        path.write_text(json.dumps(trace) + "\n")
        self.index_path.unlink()
        output = self.root / "rescore"
        with patch.object(runner, "preflight", side_effect=AssertionError("no preflight")), patch.object(runner, "load_backend", side_effect=AssertionError("no load")):
            self.assertEqual(self.main(["--evaluate-only", str(path), "--output-dir", str(output)]), 0)
        self.assertEqual(json.loads((output / "metrics.json").read_text())["strict_id"]["selected_recovered"], 1)

    def test_existing_or_protected_outputs_cannot_be_overwritten(self):
        sentinel = self.root / "existing"
        sentinel.mkdir()
        (sentinel / "keep").write_text("keep")
        with self.assertRaises(FileExistsError):
            runner.new_output_dir(sentinel)
        with self.assertRaises(ValueError):
            runner.new_output_dir(runner.ROOT / "results/preliminary/new_enhancement")
        self.assertEqual((sentinel / "keep").read_text(), "keep")

    def test_import_does_not_load_models_or_shared_team_modules(self):
        code = '''
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in ('torch', 'faiss', 'numpy', 'sentence_transformers') or name.startswith(('src.generation', 'src.diagnosis', 'src.controller', 'src.probe')):
        raise AssertionError(name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from scripts.run_nq_retrieval_enhancement import parser
assert parser().parse_args([]).device == 'cpu'
'''
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=runner.ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
