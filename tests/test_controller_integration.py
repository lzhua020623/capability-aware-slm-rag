"""Cross-module execution, leakage, failure accounting and CLI checks; no GPU."""

import contextlib
from dataclasses import asdict, replace
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.run_controller_integration import main, read_inputs, runtime_input, smoke_inputs, summarize
from src.compensation.answer_repair import Generation, MODEL_NAME
from src.compensation.retrieval_enhancement.p3 import RERANKER_MODEL, RERANKER_REVISION
from src.controller.controller import decide_action
from src.controller.execution import CallLog, ExecutionConfig, IntegratedController
from src.controller.providers import shared_repair_views
from src.controller.schema import ControllerAction, ControllerDecision
from src.diagnosis.hierarchical_classifier import HierarchicalDiagnosis
from src.diagnosis.labels import DiagnosisLabel
from src.probe.schema import ProbeResult


class Generator:
    def __init__(self, text="Amber"):
        self.text, self.calls = text, []

    def generate(self, prompt, max_new_tokens):
        self.calls.append((prompt, max_new_tokens))
        if isinstance(self.text, Exception):
            raise self.text
        return Generation(self.text, 0.01, 12, 3)


class Scorer:
    model_name, revision = RERANKER_MODEL, RERANKER_REVISION
    def __init__(self, reverse=True):
        self.reverse, self.calls = reverse, 0

    def __call__(self, query, texts):
        self.calls += 1
        return [float(i if self.reverse else -i) for i in range(len(texts))]


class Retriever:
    def __init__(self):
        self.calls = 0

    def retrieve(self, query, *, dataset, top_k):
        self.calls += 1
        return [{"passage_id": f"extra{i}", "text": "The capital of Luma is Amber.",
                 "score": 1.0 - i / 100} for i in range(top_k)]


def diagnosis(label="correct", p=None, low=False):
    label = DiagnosisLabel(label)
    p = (0.1 if label == DiagnosisLabel.CORRECT else 0.9) if p is None else p
    scores = None if label == DiagnosisLabel.CORRECT else {
        DiagnosisLabel.EVIDENCE_INSUFFICIENCY: 0.4 if low else 0.05,
        DiagnosisLabel.EVIDENCE_ACCESSIBILITY: 0.35 if low else 0.05,
        DiagnosisLabel.EVIDENCE_UTILISATION: 0.25 if low else 0.9}
    return HierarchicalDiagnosis(p, "correct" if label == DiagnosisLabel.CORRECT else "failure",
                                 scores, label, 0.9, None)


def pipeline(*, label="correct", raw="Amber", repair=None, config=None, diag=None, scorer=None):
    gen = Generator(raw)
    if repair is None:
        repair = "RANK: 1\nQUOTE: The capital of Luma is Amber.\nANSWER: Amber"
    return IntegratedController(SimpleNamespace(diagnose=lambda _: diag or diagnosis(label)), gen,
                                {"nq": Generator(repair), "fever": Generator(repair)},
                                scorer or Scorer(), Retriever(), config=config or ExecutionConfig())


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.nq, self.fever = smoke_inputs()

    def test_correct_route_has_zero_provider_calls(self):
        r = pipeline().run(self.nq)
        self.assertEqual(r["status"], "no_action")
        self.assertFalse(r["executed"])
        self.assertEqual(r["calls"], [])
        self.assertEqual(r["final_answer"], self.nq.initial_answer)
        self.assertEqual(r["cost"]["input_tokens"], 0)

    def test_routes_execute_real_modules_and_preserve_inputs(self):
        before = asdict(self.nq)
        for label, expected in (("evidence_insufficiency", (1, 1, 1)),
                                ("evidence_accessibility", (0, 1, 1)),
                                ("evidence_utilisation", (0, 0, 1))):
            with self.subTest(label=label):
                r = pipeline(label=label).run(self.nq)
                self.assertEqual(r["final_answer"], "Amber")
                self.assertTrue(r["executed"] and r["answer_changed"])
                self.assertEqual(tuple(r["cost"][k] for k in ("retrieval_calls", "scoring_calls", "llm_calls")), expected)
                self.assertIsNotNone(r["module_trace"])
                self.assertEqual(asdict(self.nq), before)
                json.dumps(r, allow_nan=False)

    def test_fever_retrieval_is_explicit_unsupported_without_providers(self):
        r = pipeline(label="evidence_insufficiency").run(self.fever)
        self.assertEqual(r["status"], "unsupported_dataset")
        self.assertFalse(r["executed"] or r["answer_changed"])
        self.assertEqual(r["calls"], [])

    def test_fever_uses_existing_qwen_repair_not_nli(self):
        repair = json.dumps({"verdict": "repair", "answer": "SUPPORTS", "citations": [
            {"rank": 1, "quote": "The capital of Luma is Amber."}]})
        r = pipeline(repair=repair).run(self.fever, mode="always_repair")
        self.assertEqual(r["final_answer"], "SUPPORTS")
        self.assertEqual(r["cost"]["llm_calls"], 1)
        self.assertEqual(r["module_trace"]["verifier_calls"], 0)

    def test_refiner_unchanged_order_spends_no_generation(self):
        r = pipeline(scorer=Scorer(reverse=False)).run(self.nq, mode="always_rerank")
        self.assertEqual(r["status"], "no_change")
        self.assertEqual((r["cost"]["scoring_calls"], r["cost"]["llm_calls"]), (1, 0))

    def test_rejected_candidate_retains_original_context(self):
        r = pipeline(raw="invented answer").run(self.nq, mode="always_rerank")
        self.assertEqual(r["status"], "kept_original")
        self.assertEqual(r["final_evidence"], [asdict(e) for e in self.nq.evidence])
        self.assertEqual(r["module_trace"]["evidence"][0]["passage_id"], "p5")

    def test_probe_safe_binary_is_unchanged_and_receipted(self):
        raw = json.dumps({"answer_supported": "supported", "evidence_sufficiency": "sufficient",
                          "evidence_accessibility": "clear", "supporting_ranks": [1], "conflicting_ranks": []})
        d = diagnosis("evidence_utilisation", p=0.6, low=True)
        r = pipeline(diag=d, raw=raw).run(self.nq)
        expected = decide_action(d, True, ProbeResult.from_judgments("supported", "sufficient", "clear", (1,), ()), self.nq.sample_id)
        self.assertEqual(r["routing"]["decision"], expected.to_dict())
        self.assertEqual(r["status"], "no_action")
        self.assertEqual(r["cost"]["llm_calls"], 1)
        self.assertEqual(r["calls"][0]["stage"], "probe")

    def test_probe_failure_keeps_type_and_counts_failed_attempt(self):
        r = pipeline(diag=diagnosis("evidence_utilisation", p=0.6, low=True),
                     raw=RuntimeError("backend failed")).run(self.nq)
        self.assertEqual(r["routing"]["decision"]["action"], "generation_repair")
        self.assertEqual((r["cost"]["llm_calls"], r["cost"]["failed_calls"]), (2, 1))
        self.assertIsNone(r["cost"]["input_tokens"])
        self.assertIsNotNone(r["routing"]["probe_error"])

    def test_budget_is_shared_by_probe_and_compensation(self):
        r = pipeline(diag=diagnosis("evidence_utilisation", p=0.6, low=True), raw="{}",
                     config=ExecutionConfig(max_llm_calls=1)).run(self.nq)
        self.assertEqual(r["status"], "budget_exceeded")
        self.assertEqual(r["cost"]["llm_calls"], 1)
        self.assertEqual(r["cost"]["budget_blocks"], 1)
        self.assertFalse(r["answer_changed"])

    def test_budget_failures_in_p3_are_not_reported_as_success(self):
        for config in (ExecutionConfig(max_llm_calls=0), ExecutionConfig(max_retrieval_calls=0),
                       ExecutionConfig(max_scoring_calls=0)):
            with self.subTest(config=config):
                r = pipeline(config=config).run(self.nq, mode="always_retrieve")
                self.assertEqual(r["status"], "budget_exceeded")
                self.assertFalse(r["answer_changed"])

    def test_generation_exception_is_an_error_and_a_counted_call(self):
        for mode in ("always_rerank", "always_retrieve", "always_repair"):
            with self.subTest(mode=mode):
                r = pipeline(raw=RuntimeError("fake OOM"), repair=RuntimeError("fake OOM")).run(self.nq, mode=mode)
                self.assertTrue(r["status"].endswith("_failed"))
                self.assertEqual(r["cost"]["failed_calls"], 1)
                self.assertEqual(r["cost"]["llm_calls"], 1)
                self.assertEqual(r["final_answer"], self.nq.initial_answer)

    def test_invalid_repair_output_is_not_success(self):
        r = pipeline(repair="not a contract").run(self.nq, mode="always_repair")
        self.assertFalse(r["answer_changed"])
        self.assertIsNotNone(r["module_trace"]["validation_error"])
        self.assertEqual(r["cost"]["llm_calls"], 1)

    def test_invalid_provider_values_do_not_break_fallback_serialization(self):
        p = pipeline()
        p.scorer = SimpleNamespace(model_name=RERANKER_MODEL, revision=RERANKER_REVISION)
        # A callable invalid scorer is counted as one failed attempt.
        class BadScorer(Scorer):
            def __call__(self, query, texts):
                return [float("nan")] * len(texts)
        p.scorer = BadScorer()
        r = p.run(self.nq, mode="always_rerank")
        self.assertEqual(r["cost"]["failed_calls"], 1)
        self.assertFalse(r["answer_changed"])
        json.dumps(r, allow_nan=False)
        p = pipeline()
        p.retriever = SimpleNamespace(retrieve=lambda *a, **kw: [{"score": float("nan")}])
        r = p.run(self.nq, mode="always_retrieve")
        self.assertEqual(r["cost"]["failed_calls"], 1)
        self.assertFalse(r["answer_changed"])
        json.dumps(r, allow_nan=False)

    def test_bad_decision_fails_before_provider_calls(self):
        p = pipeline()
        for d in (ControllerDecision("generation_repair", "correct", 0.9, False, False, self.nq.sample_id),
                  ControllerDecision("no_action", "correct", 0.9, False, False, "another-id")):
            with self.assertRaises(ValueError):
                p.execute(self.nq, d)
        self.assertEqual(p.raw_generator.calls, [])

    def test_config_rejects_unsupported_or_wrong_dataset_strategy(self):
        for kwargs in ({"fever_repair_strategy": "short_answer_quote"}, {"nq_repair_strategy": "fever_nli"},
                       {"max_llm_calls": True}, {"repair_max_new_tokens": 0}):
            with self.assertRaises(ValueError):
                ExecutionConfig(**kwargs)

    def test_shared_views_do_not_load_or_copy_model_weights(self):
        raw = SimpleNamespace(model=object(), tokenizer=object(), json_output=False)
        views = shared_repair_views(raw, ExecutionConfig())
        self.assertIs(views["nq"].model, views["fever"].model)
        self.assertIs(views["nq"].tokenizer, raw.tokenizer)
        self.assertFalse(views["nq"].json_output or raw.json_output)
        self.assertTrue(views["fever"].json_output)

    def test_summaries_separate_datasets_and_do_not_claim_accuracy(self):
        rows = [pipeline().run(i, mode="c1") for i in (self.nq, self.fever)]
        summary = summarize(rows)
        self.assertEqual(set(summary["groups"]), {"nq/c1", "fever/c1"})
        self.assertIn("not_scored", summary["quality_metrics"])


class InputAndRunnerTests(unittest.TestCase):
    def c1(self):
        item = smoke_inputs()[0]
        return {"condition": "C1", "model_name": MODEL_NAME, "dataset": "natural_questions",
                "sample_id": item.sample_id, "question_or_claim": item.query,
                "prediction": item.initial_answer, "evidence": [asdict(e) for e in item.evidence],
                "retrieved_top5_ids": [e.passage_id for e in item.evidence]}

    def test_offline_fields_cannot_change_runtime_or_prompts(self):
        row = self.c1()
        clean = runtime_input(row)
        row.update(gold_answer_or_label="GOLD_SENTINEL", target_label="correct", correct=True,
                   gold_evidence_in_top5=True, metadata={"answer": "PRIVATE_SENTINEL"})
        self.assertEqual(runtime_input(row), clean)
        result = pipeline(label="evidence_utilisation").run(runtime_input(row))
        self.assertNotIn("SENTINEL", json.dumps(result))

    def test_raw_runtime_schema_rejects_offline_fields(self):
        row = asdict(smoke_inputs()[0])
        row["gold_answer"] = "Amber"
        with self.assertRaises(ValueError):
            runtime_input(row)

    def test_duplicate_ids_and_malformed_top5_rejected(self):
        row = asdict(smoke_inputs()[0])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.jsonl"
            path.write_text((json.dumps(row) + "\n") * 2)
            with self.assertRaises(ValueError):
                read_inputs(path)
        row["evidence"] = row["evidence"][:4]
        with self.assertRaises(ValueError):
            runtime_input(row)

    def test_smoke_writes_real_receipts_and_rejects_overwrite(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            out = Path(directory) / "run"
            self.assertEqual(main(["--smoke", "--output-dir", str(out)]), 0)
            manifest = json.loads((out / "manifest.json").read_text())
            self.assertEqual((manifest["state"], manifest["n_records"]), ("completed", 10))
            rows = [json.loads(line) for line in (out / "predictions.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 10)
            unsupported = [r for r in rows if r["status"] == "unsupported_dataset"]
            self.assertEqual(len(unsupported), 1)
            self.assertEqual(unsupported[0]["calls"], [])
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(["--smoke", "--output-dir", str(out)])

    def test_preflight_needs_no_models_or_index_and_writes_no_predictions(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            inp, out = Path(directory) / "input.jsonl", Path(directory) / "run"
            inp.write_text(json.dumps(self.c1()) + "\n")
            self.assertEqual(main(["--input", str(inp), "--output-dir", str(out), "--preflight"]), 0)
            self.assertFalse((out / "predictions.jsonl").exists())
            manifest = json.loads((out / "manifest.json").read_text())
            self.assertTrue(manifest["nq_index_required"])

    def test_c1_mode_needs_no_gpu_or_index(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            inp, out = Path(directory) / "input.jsonl", Path(directory) / "run"
            inp.write_text(json.dumps(self.c1()) + "\n")
            self.assertEqual(main(["--input", str(inp), "--output-dir", str(out), "--mode", "c1"]), 0)
            record = json.loads((out / "predictions.jsonl").read_text())
            self.assertEqual(record["cost"]["llm_calls"], 0)

    def test_real_nq_adaptive_requires_explicit_index_before_loading_models(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stderr(io.StringIO()):
            inp, out = Path(directory) / "input.jsonl", Path(directory) / "run"
            inp.write_text(json.dumps(self.c1()) + "\n")
            with self.assertRaises(SystemExit):
                main(["--input", str(inp), "--output-dir", str(out)])
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
