"""Behavior, leakage boundaries, paired metrics, and resumability for E. No GPU."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

from scripts.run_answer_repair import main, prepare_records, route_status, runtime_input
from src.compensation.answer_repair import Generation, MODEL_NAME, repair_answer
from src.compensation.qwen_repair import complete_json_object
from src.compensation.controller_adapter import generation_repair
from src.diagnosis import DiagnosisInput, DiagnosisOutput, EvidenceItem
from src.evaluation.answer_repair import summarize_repairs


def request(dataset="nq", answer="London", text="The capital of France is Paris."):
    return DiagnosisInput("1", dataset, "What is the capital of France?", [EvidenceItem(1, "p1", text, 0.8)], answer)


def verdict(kind="repair", answer="Paris", quote="The capital of France is Paris.", rank=1):
    return json.dumps({"verdict": kind, "answer": answer, "citations": [{"rank": rank, "quote": quote}]})


class FakeGenerator:
    def __init__(self, text):
        self.text, self.calls = text, []

    def generate(self, prompt, max_new_tokens):
        self.calls.append((prompt, max_new_tokens))
        return Generation(self.text, 0.25, 100, 40)


def c1_row(sample_id="1", prediction="London"):
    return {
        "sample_id": sample_id, "dataset": "natural_questions", "condition": "C1",
        "model_name": MODEL_NAME, "question_or_claim": "What is the capital of France?",
        "evidence": [{"passage_id": "p1", "text": "The capital of France is Paris.", "score": 0.8}],
        "prediction": prediction, "gold_answer_or_label": "Paris", "correct": prediction == "Paris",
    }


class RepairBehaviorTests(unittest.TestCase):
    def test_one_call_grounded_nq_repair(self):
        generator = FakeGenerator(verdict())
        result = repair_answer(request(), generator)
        self.assertEqual((result.final_answer, result.status, result.extra_calls), ("Paris", "repaired", 1))
        self.assertEqual(result.citations[0].passage_id, "p1")
        self.assertEqual(len(generator.calls), 1)
        self.assertEqual(result.independent_support_status, "not_checked")

    def test_keep_preserves_original_including_whitespace(self):
        item = request(answer=" Paris ")
        result = repair_answer(item, FakeGenerator(verdict("keep", "Paris")))
        self.assertEqual(result.final_answer, " Paris ")
        self.assertFalse(result.changed)

    def test_insufficient_preserves_original(self):
        text = json.dumps({"verdict": "insufficient", "answer": "", "citations": []})
        result = repair_answer(request(), FakeGenerator(text))
        self.assertEqual((result.final_answer, result.status), ("London", "insufficient_evidence"))

    def test_fever_repair_requires_exact_label(self):
        item = request("fever", "SUPPORTS", "The capital is Paris, not London.")
        good = verdict("repair", "REFUTES", "The capital is Paris, not London.")
        self.assertEqual(repair_answer(item, FakeGenerator(good)).final_answer, "REFUTES")
        bad = verdict("repair", "REFUTES or SUPPORTS", "The capital is Paris, not London.")
        self.assertEqual(repair_answer(item, FakeGenerator(bad)).status, "invalid_output")

    def test_hallucinated_quotes_and_ranks_fall_back(self):
        for text in (verdict(quote="Paris was founded in 2000."), verdict(rank=9), verdict(rank=True)):
            with self.subTest(text=text):
                result = repair_answer(request(), FakeGenerator(text))
                self.assertEqual((result.status, result.final_answer), ("invalid_output", "London"))

    def test_citation_spacing_is_allowed_but_words_and_punctuation_are_preserved(self):
        item = request(text="The capital is Paris , not London .")
        good = verdict(quote="The capital is Paris, not London.")
        self.assertEqual(repair_answer(item, FakeGenerator(good)).status, "repaired")
        for quote in ("The capital is Paris not London.", "The capital is Paris, London.",
                      "The capital is Berlin, not London."):
            self.assertEqual(repair_answer(item, FakeGenerator(verdict(quote=quote))).status, "invalid_output")

    def test_quotes_cannot_turn_subwords_into_answer_evidence(self):
        item = request(text="Yorkshire is a county.")
        self.assertEqual(repair_answer(item, FakeGenerator(verdict(answer="York", quote="York"))).status,
                         "invalid_output")
        item = request(text="NewYork is written as one word.")
        self.assertEqual(repair_answer(item, FakeGenerator(verdict(answer="New York", quote="New York"))).status,
                         "invalid_output")

    def test_json_stop_handles_escaped_braces_and_partial_objects(self):
        for text in ('', '{"answer":"unfinished', '[{}]', 'SUPPORTS', '```json\\n{}'):
            self.assertFalse(complete_json_object(text))
        self.assertTrue(complete_json_object(json.dumps({"answer": 'a } and an escaped " quote', "citations": []})))
        self.assertTrue(complete_json_object('{"verdict":"keep"}'))

    def test_nq_answer_must_be_a_complete_span_in_citation(self):
        for item, text in (
            (request(), verdict(answer="Berlin")),
            (request(text="Yorkshire is a county."), verdict(answer="York", quote="Yorkshire is a county.")),
        ):
            self.assertEqual(repair_answer(item, FakeGenerator(text)).status, "invalid_output")

    def test_malformed_and_duplicate_key_json_do_not_change_answer(self):
        cases = ["not JSON", '{"verdict":"repair",',
                 '{"verdict":"keep","verdict":"repair","answer":"Paris","citations":[]}',
                 '{"verdict":"repair","answer":"Paris","citations":[],"gold_answer":"Paris"}']
        for text in cases:
            result = repair_answer(request(), FakeGenerator(text))
            self.assertEqual((result.status, result.final_answer), ("invalid_output", "London"))

    def test_json_fence_is_accepted(self):
        result = repair_answer(request(), FakeGenerator("```json\n" + verdict() + "\n```"))
        self.assertEqual(result.final_answer, "Paris")

    def test_keep_cannot_silently_change_answer(self):
        self.assertEqual(repair_answer(request(), FakeGenerator(verdict("keep"))).status, "invalid_output")

    def test_repair_cannot_relabel_unchanged_answer_as_a_repair(self):
        item = request(answer="Paris")
        result = repair_answer(item, FakeGenerator(verdict()))
        self.assertEqual((result.status, result.final_answer), ("invalid_output", "Paris"))

    def test_observed_explanatory_insufficient_and_trailing_text_are_rejected(self):
        for text in (
            '{"verdict":"insufficient","answer":"No evidence about this entity.","citations":[]}',
            verdict() + "````\\nAssistant justification: changed.",
        ):
            result = repair_answer(request(), FakeGenerator(text))
            self.assertEqual((result.status, result.final_answer), ("invalid_output", "London"))

    def test_empty_evidence_makes_zero_calls(self):
        item = DiagnosisInput("1", "nq", "q", [], "original")
        generator = FakeGenerator(verdict())
        self.assertEqual(repair_answer(item, generator).extra_calls, 0)
        self.assertEqual(generator.calls, [])

    def test_reanswer_control_is_distinct(self):
        generator = FakeGenerator("Paris")
        result = repair_answer(request(), generator, strategy="reanswer")
        self.assertEqual((result.final_answer, result.status, result.citations), ("Paris", "reanswered", []))

    def test_grounded_candidate_hides_initial_answer_and_derives_verdict(self):
        generator = FakeGenerator(json.dumps({"answer":"Paris", "citations":[{"rank":1,"quote":"The capital of France is Paris."}]}))
        item = request(answer="WRONG_INITIAL_SENTINEL")
        result = repair_answer(item, generator, strategy="grounded_reanswer")
        self.assertNotIn("WRONG_INITIAL_SENTINEL", generator.calls[0][0])
        self.assertEqual((result.status, result.final_answer, result.verdict), ("repaired", "Paris", "repair"))
        result = repair_answer(request(answer="Paris"), generator, strategy="grounded_reanswer")
        self.assertEqual((result.status, result.final_answer), ("kept", "Paris"))

    def test_grounded_candidate_cannot_skip_quote_checks_or_return_extras(self):
        for value in ({"answer":"Paris", "citations":[]},
                      {"answer":"Paris", "citations":[{"rank":1,"quote":"The capital is Berlin."}]},
                      {"answer":"Paris", "citations":[], "verdict":"repair"}):
            result = repair_answer(request(), FakeGenerator(json.dumps(value)), strategy="grounded_reanswer")
            self.assertEqual((result.status, result.final_answer), ("invalid_output", "London"))
        result = repair_answer(request(), FakeGenerator('{"answer":"","citations":[]}'), strategy="grounded_reanswer")
        self.assertEqual((result.status, result.final_answer), ("insufficient_evidence", "London"))

    def test_framed_generation_preserves_raw_text_and_strict_validation(self):
        generator = FakeGenerator(verdict())
        raw = verdict() + "`"
        generator.generate = unittest.mock.Mock(return_value=Generation(verdict(), 0.25, raw_text=raw))
        result = repair_answer(request(), generator)
        self.assertEqual(result.raw_output, raw)
        self.assertEqual(result.validation_input, verdict())
        self.assertEqual(result.status, "repaired")

    def test_backend_failure_propagates(self):
        generator = FakeGenerator(verdict())
        with patch.object(generator, "generate", side_effect=RuntimeError("GPU failed")):
            with self.assertRaisesRegex(RuntimeError, "GPU failed"):
                repair_answer(request(), generator)


class AdapterAndEvaluationTests(unittest.TestCase):
    def test_resolved_controller_decisions_execute_only_e(self):
        def decision(action, label, sample_id="1"):
            return SimpleNamespace(action=action, diagnosis_label=label, sample_id=sample_id,
                diagnosis_confidence=0.1, used_probe=True, probe_overrode_diagnosis=False,
                needs_compensation=action != "no_action")
        generator = FakeGenerator(verdict())
        for action, label in (("no_action", "correct"), ("retrieval_enhancement", "evidence_insufficiency"),
                              ("evidence_refinement", "evidence_accessibility")):
            result = generation_repair(request(), decision(action, label), generator)
            self.assertEqual((result.status, result.extra_calls), ("skipped_controller_action", 0))
        self.assertEqual(generator.calls, [])
        result = generation_repair(request(), decision("generation_repair", "evidence_utilisation"), generator)
        self.assertEqual((result.final_answer, result.extra_calls), ("Paris", 1))
        self.assertEqual(len(generator.calls), 1)
        empty = DiagnosisInput("1", "nq", "query", [], "initial")
        with patch("src.compensation.qwen_repair.QwenRepairGenerator", side_effect=AssertionError("Loaded model")):
            self.assertEqual(generation_repair(empty, decision("generation_repair", "evidence_utilisation")).extra_calls, 0)
        for invalid in (decision("generation_repair", "correct"),
                        decision("generation_repair", "evidence_utilisation", sample_id="wrong")):
            with self.assertRaises(ValueError):
                generation_repair(request(), invalid, generator)
        self.assertEqual(len(generator.calls), 1)
    def test_offline_fields_never_reach_prompt(self):
        row = c1_row()
        for key in ("gold_answer_or_label", "gold_evidence_text", "c3_result", "larger_model_result", "judge_verdict"):
            row[key] = "FORBIDDEN_SENTINEL_" + key
        row["gold_evidence_in_top5"] = True
        generator = FakeGenerator(verdict())
        repair_answer(runtime_input(row, "nq"), generator)
        self.assertNotIn("FORBIDDEN_SENTINEL", generator.calls[0][0])

    def test_c3_source_and_duplicate_ids_are_rejected(self):
        row = c1_row()
        row["condition"] = "C3"
        with self.assertRaises(ValueError):
            prepare_records([row], "nq")
        with self.assertRaises(ValueError):
            prepare_records([c1_row(), c1_row()], "nq")

    def test_routing_respects_other_labels_and_pending_probe(self):
        for label in ("correct", "evidence_insufficiency", "evidence_accessibility"):
            decision = DiagnosisOutput(label, 0.9, False)
            self.assertEqual(route_status(decision, False), "skipped_other_diagnosis")
        self.assertEqual(route_status(DiagnosisOutput("evidence_utilisation", 0.9, True), False), "skipped_pending_probe")
        self.assertIsNone(route_status(DiagnosisOutput("evidence_utilisation", 0.9, False), False))
        with self.assertRaises(ValueError):
            route_status(None, False)

    def test_net_gain_counts_regressions(self):
        rows = []
        for initial, final in ((False, True), (True, False), (True, True)):
            rows.append({"initial_metrics": {"correct": initial}, "final_metrics": {"correct": final},
                         "changed": initial != final, "extra_calls": 1, "latency": 0.25, "status": "repaired"})
        summary = summarize_repairs(rows)
        self.assertEqual((summary["fixed"], summary["regressed"], summary["net_correct_gain"]), (1, 1, 0))
        self.assertEqual(summary["regression_rate"], 0.5)
        self.assertEqual(summary["independent_support_checked"], 0)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source, self.out = self.root / "c1.jsonl", self.root / "e.jsonl"
        self.source.write_text("\n".join(json.dumps(c1_row(str(i))) for i in (1, 2)) + "\n")
        self.args = ["--dataset", "nq", "--c1", str(self.source), "--all", "--out", str(self.out)]

    def invoke(self, args):
        with contextlib.redirect_stdout(io.StringIO()):
            return main(args)

    def test_preflight_never_loads_model_or_writes_predictions(self):
        with patch("src.compensation.qwen_repair.QwenRepairGenerator", side_effect=AssertionError("Loaded model")):
            self.assertEqual(self.invoke(self.args + ["--check-only"]), 0)
            self.assertEqual(self.invoke(self.args + ["--dry-run"]), 0)
        self.assertFalse(self.out.exists())

    def test_interruption_resume_missing_newline_and_completed_run(self):
        first = FakeGenerator(verdict())
        first.generate = unittest.mock.Mock(side_effect=[Generation(verdict(), 0.25), RuntimeError("interrupted")])
        with patch("src.compensation.qwen_repair.QwenRepairGenerator", return_value=first):
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                self.invoke(self.args)
        self.assertEqual(len(self.out.read_text().splitlines()), 1)
        self.assertFalse(self.out.with_suffix(".jsonl.lock").exists())
        self.out.write_text(self.out.read_text().rstrip("\n"))
        second = FakeGenerator(verdict())
        with patch("src.compensation.qwen_repair.QwenRepairGenerator", return_value=second):
            self.assertEqual(self.invoke(self.args), 0)
        self.assertEqual(len(second.calls), 1)
        rows = [json.loads(line) for line in self.out.read_text().splitlines()]
        self.assertEqual([row["sample_id"] for row in rows], ["1", "2"])
        self.assertEqual({row["condition"] for row in rows}, {"E"})
        with patch("src.compensation.qwen_repair.QwenRepairGenerator", side_effect=AssertionError("Loaded model")):
            self.assertEqual(self.invoke(self.args), 0)

    def test_changed_protocol_and_truncated_output_refuse_resume(self):
        with patch("src.compensation.qwen_repair.QwenRepairGenerator", return_value=FakeGenerator(verdict())):
            self.invoke(self.args)
        original = self.out.read_bytes()
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            self.invoke(self.args + ["--max-new-tokens", "128", "--check-only"])
        self.assertEqual(self.out.read_bytes(), original)
        self.out.write_bytes(original + b'{"sample_id":')
        with self.assertRaisesRegex(ValueError, "Invalid JSON"):
            self.invoke(self.args)
        self.assertEqual(self.out.read_bytes(), original + b'{"sample_id":')

    def test_resume_refuses_changed_runtime_model_revision(self):
        first = FakeGenerator(verdict())
        first.runtime_environment = {"model_revision": "revision-a"}
        first.generate = unittest.mock.Mock(side_effect=[Generation(verdict(), 0.25), RuntimeError("interrupted")])
        with patch("src.compensation.qwen_repair.QwenRepairGenerator", return_value=first):
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                self.invoke(self.args)
        original = self.out.read_bytes()
        self.assertEqual(json.loads(original)["runtime_environment"], first.runtime_environment)
        second = FakeGenerator(verdict())
        second.runtime_environment = {"model_revision": "revision-b"}
        with patch("src.compensation.qwen_repair.QwenRepairGenerator", return_value=second):
            with self.assertRaisesRegex(ValueError, "Runtime environment/model revision differs"):
                self.invoke(self.args)
        self.assertEqual(second.calls, [])
        self.assertEqual(self.out.read_bytes(), original)

    def test_unrouted_samples_do_not_load_model(self):
        diagnoses = self.root / "diagnoses.jsonl"
        diagnoses.write_text("\n".join(json.dumps({"sample_id": str(i), "dataset": "nq", "diagnosis_label": "correct",
                                                  "confidence": 0.9, "needs_probe": False}) for i in (1, 2)))
        args = ["--dataset", "nq", "--c1", str(self.source), "--diagnoses", str(diagnoses), "--out", str(self.out)]
        with patch("src.compensation.qwen_repair.QwenRepairGenerator", side_effect=AssertionError("Loaded model")):
            self.assertEqual(self.invoke(args), 0)
        summary = json.loads(self.out.with_suffix(".summary.json").read_text())
        self.assertEqual((summary["evaluated_samples"], summary["extra_calls"]), (2, 0))

    def test_controller_file_is_runtime_only_and_missing_decisions_fail(self):
        path = self.root / "controller.jsonl"
        rows = [{"sample_id":str(i), "action":"no_action", "diagnosis_label":"correct",
                 "diagnosis_confidence":0.9, "used_probe":False, "probe_overrode_diagnosis":False,
                 "needs_compensation":False} for i in (1,2)]
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        args = ["--dataset","nq","--c1",str(self.source),"--controller-decisions",str(path),"--out",str(self.out)]
        with patch("src.compensation.qwen_repair.QwenRepairGenerator", side_effect=AssertionError("Loaded model")):
            self.assertEqual(self.invoke(args),0)
        self.assertEqual(json.loads(self.out.with_suffix('.summary.json').read_text())["routing"],"controller")
        rows[0]['gold_answer_or_label']='Paris'
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        with self.assertRaisesRegex(ValueError,"runtime fields"):
            self.invoke(args+['--check-only'])
        del rows[0]['gold_answer_or_label']
        path.write_text(json.dumps(rows[0])+'\n')
        with self.assertRaisesRegex(ValueError,"Missing resolved ControllerDecision"):
            self.invoke(args+['--check-only'])


if __name__ == "__main__":
    unittest.main()
