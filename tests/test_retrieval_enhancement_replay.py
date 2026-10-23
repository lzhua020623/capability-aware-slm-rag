"""Replay engineering tests use synthetic records; no frozen data committed."""

import contextlib
from copy import deepcopy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from scripts import run_nq_retrieval_enhancement as runner
from src.compensation.retrieval_enhancement import EnhancementBudget
from src.compensation.retrieval_enhancement.evaluation import evaluate_nq
from src.compensation.retrieval_enhancement.replay import rewrite_coverage, select_replay, validate_replay


def baseline(sid, query="who wrote Book X"):
    return {"sample_id": sid, "dataset": "nq", "question": query,
            "base_evidence": [{"passage_id": pid, "text": f"Evidence {pid}", "score": score}
                              for pid, score in zip("ABCDE", [0.9, 0.8, 0.7, 0.6, 0.5])]}


class AdditionalOnly:
    def __init__(self):
        self.calls = []

    def retrieve(self, query, *, dataset, top_k):
        if not query.endswith(("author", "birthplace", "date of birth")):
            raise AssertionError("Original query must never retrieve in Replay")
        self.calls.append((query, dataset, top_k))
        return [{"passage_id": "F", "text": "Synthetic extra evidence", "score": 0.95},
                {"passage_id": "A", "text": "Evidence A", "score": 0.85}]


class NeverRetrieve:
    def retrieve(self, *args, **kwargs):
        raise AssertionError("No-rewrite smoke must not retrieve")


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.records = [baseline("9007199254740993123"), baseline("-9007199254740993123", "unsupported query"), baseline("dev")]
        self.manifest = {"dataset": "natural_questions", "n": 2, "seed": 42,
                         "ids": [item["sample_id"] for item in self.records[:2]]}
        self.input = self.root / "replay.jsonl"
        self.manifest_path = self.root / "manifest.json"
        self.save_records()
        self.manifest_path.write_text(json.dumps(self.manifest))
        self.arguments = ["--baseline-replay", str(self.input), "--manifest", str(self.manifest_path),
                          "--cache-dir", str(self.root / "missing_cache"), "--index", str(self.root / "missing_index"),
                          "--metadata", str(self.root / "missing_meta"), "--queries", str(self.root / "missing_queries")]

    def save_records(self):
        self.input.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in self.records))

    def main(self, extra=()):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return runner.main([*self.arguments, *extra])

    def test_validation_preserves_large_string_ids_scores_text_and_order(self):
        self.records[0]["base_evidence"][0]["text"] = "Literal \\u0391\n  original whitespace "
        original = deepcopy(self.records)
        validated, audit = validate_replay(self.records)
        self.assertEqual(validated, original)
        self.assertEqual(self.records, original)
        self.assertEqual(audit["provenance_status"], "unverified")
        validated[0]["base_evidence"][0]["text"] = "changed output"
        self.assertEqual(self.records, original)

    def test_numeric_or_duplicate_sample_ids_are_rejected(self):
        for value in (9007199254740993123, 9.007199254740993e18, True, ""):
            bad = deepcopy(self.records)
            bad[0]["sample_id"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "sample_id"):
                validate_replay(bad)
        with self.assertRaisesRegex(ValueError, "duplicate Replay"):
            validate_replay(self.records + self.records[:1])

    def test_gold_answers_em_labels_and_extra_fields_are_rejected_at_both_levels(self):
        for key in ("gold_passage_id", "gold_evidence", "gold_answer", "em_reference", "primary_label", "c1_prediction"):
            for nested in (False, True):
                bad = deepcopy(self.records[0])
                (bad["base_evidence"][0] if nested else bad)[key] = "offline only"
                backend = AdditionalOnly()
                with self.subTest(key=key, nested=nested), self.assertRaises(ValueError):
                    runner.run_replay_query(bad, backend, EnhancementBudget())
                self.assertEqual(backend.calls, [])

    def test_invalid_evidence_is_rejected_without_reordering_or_truncating(self):
        mutations = [lambda r: r["base_evidence"].pop(),
                     lambda r: r["base_evidence"].append(deepcopy(r["base_evidence"][0])),
                     lambda r: r["base_evidence"].reverse(),
                     lambda r: r["base_evidence"][1].update(passage_id="A"),
                     lambda r: r["base_evidence"][0].update(text=" "),
                     lambda r: r["base_evidence"][0].update(score=float("nan")),
                     lambda r: r["base_evidence"][0].update(score=float("inf")),
                     lambda r: r["base_evidence"][0].update(score=10 ** 1000),
                     lambda r: r["base_evidence"][0].update(score=True),
                     lambda r: r["base_evidence"][0].update(score="0.9"),
                     lambda r: r.update(dataset="fever")]
        for mutate in mutations:
            record = deepcopy(self.records[0])
            mutate(record)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                validate_replay([record])

    def test_equal_scores_keep_input_order(self):
        record = deepcopy(self.records[0])
        for item in record["base_evidence"]:
            item["score"] = 0.5
        self.assertEqual(validate_replay([record])[0][0], record)

    def test_manifest_requires_complete_exact_prefix_even_for_limit_one(self):
        for records in (self.records[:1], list(reversed(self.records[:2])), self.records[1:],
                        [self.records[0], self.records[2], self.records[1]]):
            with self.subTest(records=records), self.assertRaises(ValueError):
                select_replay(records, self.manifest, limit=1)
        self.assertEqual(select_replay(self.records, self.manifest, limit=1), self.records[:1])

    def test_development_scope_is_disjoint_and_not_label_selected(self):
        self.assertEqual(select_replay(self.records, self.manifest, scope="development"), self.records[2:])
        self.assertEqual(select_replay(self.records[2:], self.manifest, scope="development"), self.records[2:])
        with self.assertRaises(ValueError):
            select_replay(self.records[:1], self.manifest, scope="development")

    def test_cross_sample_unicode_variants_are_audited_without_decoding(self):
        self.records[0]["base_evidence"][0]["text"] = "Greek Α"
        self.records[1]["base_evidence"][0]["text"] = "Greek \\u0391"
        original = deepcopy(self.records)
        records, audit = validate_replay(self.records)
        self.assertEqual(records, original)
        self.assertEqual(len(audit["conflicts"]), 1)
        self.assertEqual(audit["conflicts"][0]["passage_id"], "A")
        self.assertTrue(audit["warnings"])

    def test_json_duplicate_fields_are_rejected(self):
        raw = json.dumps(self.records[0])
        self.input.write_text(raw.replace('"sample_id":', '"sample_id":"duplicate", "sample_id":') + "\n")
        report = runner.preflight(runner.parser().parse_args(self.arguments))
        self.assertFalse(report["replay_format_ready"])
        self.assertIn("duplicate JSON field", report["errors"][0])

    def test_format_preflight_succeeds_without_backend_but_reports_run_not_ready(self):
        with patch.object(runner, "load_backend", side_effect=AssertionError("no load")):
            self.assertEqual(self.main(), 0)
        report = runner.preflight(runner.parser().parse_args(self.arguments))
        self.assertTrue(report["replay_format_ready"])
        self.assertFalse(report["structure_ready"])
        self.assertTrue(report["additional_retrieval_required"])

    def test_zero_budget_cli_replay_needs_no_resources_and_preserves_base(self):
        output = self.root / "zero"
        original_bytes = self.input.read_bytes()
        with patch.object(runner, "load_backend", side_effect=AssertionError("no load")), patch.object(runner, "_retrieval_preflight", side_effect=AssertionError("no resources")):
            self.assertEqual(self.main(["--run", "--max-retrieval-calls", "0", "--output-dir", str(output)]), 0)
        records = runner.load_jsonl(output / "traces.jsonl")
        for original, record in zip(self.records, records):
            self.assertEqual(record["base_evidence"], original["base_evidence"])
            self.assertEqual(record["enhanced_evidence"], original["base_evidence"])
            self.assertIsNone(record["base_latency"])
            self.assertEqual(record["base_retrieval_calls"], 0)
            self.assertEqual(record["trace"]["status"], "budget_exhausted")
            self.assertEqual(record["trace"]["retrieval_calls"], 0)
            self.assertEqual(record["baseline_provenance"]["status"], "unverified")
        self.assertEqual(self.input.read_bytes(), original_bytes)
        config = json.loads((output / "run_config.json").read_text())
        self.assertEqual(config["baseline_mode"], "replay")
        self.assertEqual(config["inputs"]["baseline_replay"]["sha256"], runner.fingerprint(self.input)["sha256"])
        self.assertFalse((output / "metrics.json").exists())
        self.assertEqual(json.loads((output / "completion.json").read_text())["gold_evaluation"], "not_run")

    def test_positive_budget_missing_backend_errors_before_creating_output(self):
        output = self.root / "blocked"
        with patch.object(runner, "load_backend") as load:
            self.assertEqual(self.main(["--run", "--output-dir", str(output)]), 2)
            load.assert_not_called()
        self.assertFalse(output.exists())
        with self.assertRaisesRegex(ValueError, "requires a loaded Backend"):
            runner.run_replay_query(self.records[0], None, EnhancementBudget())

    def test_no_rewrite_engineering_path_with_positive_budget_never_calls_backend(self):
        record = self.records[1]
        result = runner.run_replay_query(record, NeverRetrieve(), EnhancementBudget())
        self.assertEqual(result["trace"]["status"], "no_rewrite")
        self.assertEqual(result["trace"]["retrieval_calls"], 0)
        self.assertEqual(result["enhanced_evidence"], record["base_evidence"])

    def test_synthetic_positive_paths_only_retrieve_rewrites_and_fuse_new_evidence(self):
        for query, rewrite in [("who wrote Book X", "Book X author"),
                               ("where was Ada Lovelace born", "Ada Lovelace birthplace"),
                               ("when was Ada Lovelace born", "Ada Lovelace date of birth")]:
            record = baseline("development", query)
            original = deepcopy(record)
            backend = AdditionalOnly()
            result = runner.run_replay_query(record, backend, EnhancementBudget(max_retrieval_calls=1))
            self.assertEqual(backend.calls, [(rewrite, "nq", 5)])
            self.assertEqual(result["trace"]["rewritten_queries"], [rewrite])
            self.assertEqual(result["trace"]["status"], "completed")
            self.assertEqual(result["trace"]["retrieval_calls"], 1)
            self.assertEqual(result["trace"]["llm_calls"], 0)
            self.assertEqual(result["trace"]["new_candidate_ids"], ["F"])
            self.assertEqual(result["trace"]["selected_ids"], list("AFBCD"))
            self.assertEqual(record, original)
            self.assertEqual(result["base_evidence"], original["base_evidence"])
            self.assertEqual(result["retriever_backend"], "injected")

    def test_retrieved_metadata_text_conflict_safely_falls_back(self):
        class Conflict:
            def retrieve(self, *args, **kwargs):
                return [{"passage_id": "A", "text": "conflicting Metadata text", "score": 0.95}]
        result = runner.run_replay_query(self.records[0], Conflict(), EnhancementBudget())
        self.assertEqual(result["trace"]["status"], "retrieval_failed")
        self.assertIn("Conflicting text", result["trace"]["errors"][0]["message"])
        self.assertEqual(result["enhanced_evidence"], self.records[0]["base_evidence"])

    def test_positive_cli_development_smoke_loads_only_additional_backend_and_releases_it(self):
        backend, owner = AdditionalOnly(), Mock()
        output = self.root / "development_positive"
        # Fingerprint tiny synthetic artifacts; never load a real FAISS Index.
        (self.root / "missing_index").write_bytes(b"SYNTHETIC TEST ONLY")
        (self.root / "missing_meta").write_text("{}\n")
        ready = {"structure_ready": True, "errors": [], "model": {"snapshot": None}}
        with patch.object(runner, "_retrieval_preflight", return_value=ready), patch.object(runner, "load_backend", return_value=(owner, backend)):
            self.assertEqual(self.main(["--run", "--replay-scope", "development", "--output-dir", str(output)]), 0)
        owner.unload.assert_called_once_with()
        self.assertEqual(backend.calls, [("Book X author", "nq", 5)])
        record = runner.load_jsonl(output / "traces.jsonl")[0]
        self.assertEqual(record["sample_id"], "dev")
        self.assertEqual(record["trace"]["status"], "completed")
        self.assertEqual(record["base_retrieval_calls"], 0)
        self.assertEqual(record["baseline_provenance"]["source"]["sha256"], runner.fingerprint(self.input)["sha256"])
        self.assertEqual(json.loads((output / "run_config.json").read_text())["scope"], "development_engineering")
        self.assertFalse((output / "metrics.json").exists())

    def test_prior_conversion_warning_is_sha_matched_but_never_verifies_provenance(self):
        audit_path = self.root / "audit.json"
        prior = {"outputs": {"input": {"sha256": runner.fingerprint(self.input)["sha256"]}},
                 "evidence": {"conflicting_ids": ["A", "B", "C", "D"]}}
        audit_path.write_text(json.dumps(prior))
        args = runner.parser().parse_args([*self.arguments, "--baseline-audit", str(audit_path)])
        audit = runner.load_replay(args)[3]
        self.assertEqual(audit["conversion_audit_conflicting_ids"], list("ABCD"))
        self.assertEqual(audit["provenance_status"], "unverified")
        prior["outputs"]["input"]["sha256"] = "wrong"
        audit_path.write_text(json.dumps(prior))
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            runner.load_replay(args)

    def test_replay_does_not_overwrite_previous_output(self):
        output = self.root / "existing"
        output.mkdir()
        sentinel = output / "keep"
        sentinel.write_text("original")
        self.assertEqual(self.main(["--run", "--max-retrieval-calls", "0", "--output-dir", str(output)]), 2)
        self.assertEqual(sentinel.read_text(), "original")
        self.assertEqual(list(output.iterdir()), [sentinel])

    def test_coverage_is_query_only_with_posthoc_initial_label_strata(self):
        records = [baseline("f1"), baseline("f2", "unsupported query"),
                   baseline("d1", "where was Ada Lovelace born"),
                   baseline("d2", "where was Ada never born"), baseline("d3", "unsupported query")]
        manifest = {"dataset": "natural_questions", "n": 2, "ids": ["f1", "f2"]}
        labels = [{"sample_id": r["sample_id"], "primary_label": "Evidence Insufficiency" if r["sample_id"] in ("f2", "d1") else "Correct / No Failure", "gold_answer": "ignored"} for r in records]
        result = rewrite_coverage(records, manifest, labels)
        self.assertEqual((result["frozen"]["n"], result["frozen"]["eligible"]), (2, 1))
        self.assertEqual((result["development"]["n"], result["development"]["eligible"]), (3, 1))
        self.assertEqual(result["all"]["reasons"], {"eligible": 2, "no_supported_template": 2, "matched_template_but_rejected_conservatively": 1})
        self.assertEqual(result["ai_initial_evidence_insufficiency"]["all"]["eligible"], 1)
        self.assertEqual(rewrite_coverage(records, manifest)["all"], result["all"])

    def test_offline_cost_does_not_invent_replayed_base_calls_or_latency(self):
        record = runner.run_replay_query(self.records[0], AdditionalOnly(), EnhancementBudget())
        metrics = evaluate_nq([record], [{"id": record["sample_id"], "gold_passage_id": "F"}])["metrics"]
        self.assertEqual(metrics["cost"]["base_retrieval_calls"], 0)
        self.assertEqual(metrics["cost"]["total_retrieval_calls"], 1)
        self.assertIsNone(metrics["cost"]["mean_base_latency_seconds"])
        self.assertEqual(metrics["cost"]["base_latency_observed_samples"], 0)
        self.assertEqual(metrics["cost"]["mean_total_latency_seconds"], record["trace"]["latency"])

    def test_zero_replay_import_and_execution_need_no_heavy_packages(self):
        code = '''
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in ('torch','faiss','numpy','sentence_transformers'):
        raise AssertionError(name)
    return original(name,*args,**kwargs)
builtins.__import__=guarded
from scripts.run_nq_retrieval_enhancement import run_replay_query
from src.compensation.retrieval_enhancement import EnhancementBudget
record={'sample_id':'9223372036854775001','dataset':'nq','question':'who wrote Book X',
        'base_evidence':[{'passage_id':str(i),'text':'text '+str(i),'score':1-i/10} for i in range(5)]}
assert run_replay_query(record,None,EnhancementBudget(max_retrieval_calls=0))['trace']['status']=='budget_exhausted'
'''
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=runner.ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
