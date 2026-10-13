"""Provenance, independent NLI abstention, budgets and runtime boundaries."""
import contextlib
import io
import tempfile
from pathlib import Path
from unittest.mock import patch
import json
import math
import re
import unittest
from dataclasses import asdict, replace
from types import SimpleNamespace

from src.compensation.answer_repair import repair_answer
from src.compensation.controller_adapter import generation_repair
from src.compensation.extractive_repair import canonical_excerpt
from src.compensation.nli_repair import (
    NLI_MODEL, NLI_REVISION, NLI_THRESHOLD, evidence_windows, repair_from_nli_trace,
)
from src.diagnosis import DatasetName, EvidenceItem
from src.evaluation.answer_repair import summarize_repairs
from tests.test_answer_repair import request, c1_row
from scripts.run_answer_repair import main, validate_runtime_environment
from tests.test_answer_repair_v5 import SequenceGenerator


def extract(spans):
    return SequenceGenerator([json.dumps({"spans": spans})])


class SpanTests(unittest.TestCase):
    def test_copies_canonical_evidence_instead_of_authored_answer(self):
        r = request(text="Most people waited in 1922 - 1991 .")
        result = repair_answer(r, extract([{"rank": 1, "quote": "most people waited in 1922-1991."}]),
                               strategy="extractive_reanswer")
        self.assertEqual(result.final_answer, r.evidence[0].text)
        self.assertEqual(result.citations[0].quote, result.final_answer)
        self.assertEqual((result.extra_calls, result.independent_support_status), (1, "not_checked"))

    def test_multipart_answer_keeps_each_part_traceable(self):
        r = request(text="Mira lives in Paris. Leon lives in Rome.")
        result = repair_answer(r, extract([{"rank": 1, "quote": "Mira lives in Paris."},
                                           {"rank": 1, "quote": "Leon lives in Rome."}]),
                               strategy="extractive_reanswer")
        self.assertEqual(result.final_answer, "Mira lives in Paris. ; Leon lives in Rome.")
        self.assertEqual(len(result.citations), 2)

    def test_cannot_drop_negation_splice_or_match_word_substrings(self):
        text = "Yorkshire is not in Paris. Rome is sunny."
        for quote in ("York", "Yorkshire is in Paris.", "Paris. is sunny.", "\"Rome is sunny.\""):
            with self.subTest(quote=quote), self.assertRaises(ValueError):
                canonical_excerpt(quote, text)

    def test_bad_schema_repetition_and_lengths_preserve_original(self):
        for spans in ([{"rank": True, "quote": "Paris"}], [{"rank": 6, "quote": "Paris"}],
                      [{"rank": 1, "quote": "Paris", "answer": "fake"}],
                      [{"rank": 1, "quote": "Paris"}] * 2,
                      [{"rank": 1, "quote": "Paris"}] * 4):
            result = repair_answer(request(), extract(spans), strategy="extractive_reanswer")
            self.assertEqual((result.status, result.final_answer), ("invalid_output", "London"))
        result = repair_answer(request(text="word " * 81),
                               extract([{"rank": 1, "quote": "word " * 81}]), strategy="extractive_reanswer")
        self.assertEqual(result.status, "invalid_output")

    def test_abstention_and_no_evidence_make_no_unearned_claim(self):
        r = repair_answer(request(), extract([]), strategy="extractive_reanswer")
        self.assertEqual((r.status, r.final_answer), ("insufficient_evidence", "London"))
        r = repair_answer(request(text=""), extract([]), strategy="extractive_reanswer")
        self.assertEqual(r.extra_calls, 0)


def fever_request():
    r = request(dataset=DatasetName.FEVER, answer="REFUTES", text="Mira is a botanist.")
    return r


def nli_trace(r, probabilities=None):
    item = r.evidence[0]
    return {"model_name": NLI_MODEL, "model_revision": NLI_REVISION, "threshold": NLI_THRESHOLD,
            "calls": 1, "latency": 0.5, "input_tokens": 30,
            "windows": [{"rank": item.rank, "passage_id": item.passage_id, "quote": item.text,
                         "start": 0, "end": len(item.text), "input_tokens": 30,
                         "probabilities": probabilities or {"ENTAILMENT": .96, "CONTRADICTION": .02, "NEUTRAL": .02}}]}


class WordTokenizer:
    """Token-length fixture, not a model or a quality-evaluation substitute."""
    def encode(self, text, **kwargs):
        return text.split()

    def num_special_tokens_to_add(self, pair):
        return 3

    def __call__(self, text, other=None, **kwargs):
        matches = list(re.finditer(r"\S+", text))
        return {"offset_mapping": [(m.start(), m.end()) for m in matches],
                "input_ids": [0] * (len(matches) + (len(other.split()) + 3 if other else 0))}


class NLITests(unittest.TestCase):
    def test_decision_and_classifier_costs_are_separate(self):
        r = fever_request()
        result = repair_from_nli_trace(r, nli_trace(r))
        self.assertEqual((result.final_answer, result.extra_calls, result.verifier_calls), ("SUPPORTS", 0, 1))
        row = {**asdict(result), "changed": True, "initial_metrics": {"correct": False},
               "final_metrics": {"correct": True}}
        summary = summarize_repairs([row])
        self.assertEqual((summary["extra_generation_latency_total"], summary["verifier_latency_total"]), (0, .5))

    def test_neutral_is_not_refutation(self):
        r = fever_request()
        trace = nli_trace(r, {"ENTAILMENT": .01, "CONTRADICTION": .01, "NEUTRAL": .98})
        result = repair_from_nli_trace(r, trace)
        self.assertEqual((result.status, result.final_answer, result.citations), ("nli_uncertain", "REFUTES", []))

    def test_contradiction_and_conflicting_windows(self):
        r = fever_request()
        trace = nli_trace(r, {"ENTAILMENT": .01, "CONTRADICTION": .98, "NEUTRAL": .01})
        self.assertEqual(repair_from_nli_trace(r, trace).independent_support_status, "nli_contradiction")
        trace["windows"].extend(nli_trace(r)["windows"])
        trace["input_tokens"] = 60
        self.assertEqual(repair_from_nli_trace(r, trace).status, "nli_conflict")

    def test_bad_model_threshold_probability_or_quote_cannot_decide(self):
        r = fever_request()
        bad = []
        for key, value in (("model_revision", "other"), ("threshold", .5), ("latency", math.nan)):
            trace = nli_trace(r); trace[key] = value; bad.append(trace)
        trace = nli_trace(r); trace["windows"][0]["quote"] = "Made up"; bad.append(trace)
        trace = nli_trace(r); trace["windows"][0]["probabilities"]["ENTAILMENT"] = math.nan; bad.append(trace)
        for trace in bad:
            with self.assertRaises(ValueError):
                repair_from_nli_trace(r, trace)

    def test_long_premises_are_covered_and_long_claim_is_not_truncated(self):
        text = ' '.join(f'word{i}' for i in range(700))
        r = request(text=text)
        windows = evidence_windows(r, WordTokenizer(), 128)
        covered = {word for w in windows for word in w['quote'].split()}
        self.assertEqual(covered, set(text.split()))
        self.assertTrue(all(w['input_tokens'] <= 128 for w in windows))
        r = replace(request(text=text), query=' '.join(['claim'] * 100))
        with self.assertRaises(ValueError):
            evidence_windows(r, WordTokenizer(), 128)

    def test_nli_only_receives_runtime_request_and_no_generator_call(self):
        r = fever_request()
        verifier = SimpleNamespace(score=lambda data: nli_trace(data))
        result = repair_answer(r, SequenceGenerator([]), strategy="fever_nli", verifier=verifier)
        self.assertEqual(result.final_answer, "SUPPORTS")
        with self.assertRaises(ValueError):
            repair_answer(request(), strategy="fever_nli", verifier=verifier)
        with self.assertRaises(ValueError):
            repair_answer(r, extract([]), strategy="extractive_reanswer")

    def test_classifier_environment_roundtrips_json_keys(self):
        env = {"id2label": {0: "CONTRADICTION", 1: "NEUTRAL", 2: "ENTAILMENT"}}
        saved = [{"runtime_environment": json.loads(json.dumps(env)), "extra_calls": 0, "verifier_calls": 1}]
        validate_runtime_environment(saved, env)
        with self.assertRaises(ValueError):
            validate_runtime_environment(saved, {"id2label": {0: "ENTAILMENT"}})

    def test_real_runner_nli_resume_and_trace_tampering(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, out = root/'input.jsonl', root/'output.jsonl'
            records = []
            for n in (1, 2):
                row = c1_row(str(n))
                row.update(dataset="fever", prediction="REFUTES", gold_answer_or_label="SUPPORTS", correct=False)
                records.append(row)
            source.write_text(''.join(json.dumps(r) + '\n' for r in records))
            args = ['--dataset', 'fever', '--c1', str(source), '--all', '--strategy', 'fever_nli', '--out', str(out)]
            count = 0
            def interrupted(data):
                nonlocal count
                count += 1
                if count == 2: raise RuntimeError('interrupted')
                return nli_trace(data)
            env = {"id2label": {0: "CONTRADICTION", 1: "NEUTRAL", 2: "ENTAILMENT"}}
            first = SimpleNamespace(score=interrupted, runtime_environment=env)
            with contextlib.redirect_stdout(io.StringIO()), patch('src.compensation.nli_repair.MNLIVerifier', return_value=first):
                with self.assertRaisesRegex(RuntimeError, 'interrupted'): main(args)
            second = SimpleNamespace(score=nli_trace, runtime_environment=env)
            with contextlib.redirect_stdout(io.StringIO()), patch('src.compensation.nli_repair.MNLIVerifier', return_value=second):
                main(args)
            rows = [json.loads(l) for l in out.read_text().splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual(sum(r['extra_calls'] for r in rows), 0)
            rows[0]['verifier_latency'] = 9
            out.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'trace/cost mismatch'):
                main(args + ['--check-only'])

    def test_upstream_no_action_does_not_load_or_call_any_backend(self):
        r = fever_request()
        decision = SimpleNamespace(action="no_action", diagnosis_label="correct", diagnosis_confidence=.9,
                                   used_probe=False, probe_overrode_diagnosis=False,
                                   needs_compensation=False, sample_id=r.sample_id)
        result = generation_repair(r, decision, strategy="fever_nli")
        self.assertEqual((result.status, result.verifier_calls), ("skipped_controller_action", 0))


if __name__ == "__main__":
    unittest.main()
