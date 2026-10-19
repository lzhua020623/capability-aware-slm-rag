"""Prospective candidate grammar compatibility, with unchanged source rejection."""
import json
import unittest
from dataclasses import asdict
from pathlib import Path

from src.compensation.answer_repair import repair_answer
from src.compensation.conservative_repair import parse_candidate, repair_from_v8_trace
from src.compensation.line_contract import (
    CANDIDATE_LABELS, FormatError, GroundingError, complete_labeled_output, parse_candidate_lines,
)
from tests.test_answer_repair import request
from tests.test_answer_repair_v5 import SequenceGenerator
from tests.test_answer_repair_v8 import WRONG, CANDIDATE, check
from tests.test_answer_repair_v8_r2 import real_request
from tests import test_answer_repair_v8_r2 as r2_tests
from scripts.check_answer_repair_v8_smoke import evaluate_smoke


class CandidateGrammarTests(unittest.TestCase):
    def test_new_gate_distinguishes_source_refusal_without_rewriting_r2_gate(self):
        manifest, rows = r2_tests.SmokeGateTests().inputs()
        stage = rows["conservative_no_original_gate"][0]["repair_trace"]["stages"][0]
        stage.update(validation_error="answer outside quote", validation_kind="grounding", parsed=None)
        self.assertFalse(evaluate_smoke(manifest, rows)["engineering_gate_passed"])
        manifest["smoke_acceptance"].pop("max_invalid_v8_stages")
        manifest["smoke_acceptance"].update(max_format_invalid_v8_stages=0,
                                             allowed_guard_rejections=["grounding", "constraint"])
        self.assertTrue(evaluate_smoke(manifest, rows)["engineering_gate_passed"])
        for kind in ("format", None):
            stage["validation_kind"] = kind
            self.assertFalse(evaluate_smoke(manifest, rows)["engineering_gate_passed"])

    def test_three_explicit_forms_return_the_same_source_answer(self):
        bare = CANDIDATE.removeprefix("RANK: ")
        positional = bare.replace("QUOTE: ", "").replace("ANSWER: ", "")
        for text, schema in ((CANDIDATE, "labeled"), (bare, "bare_rank_labeled_values"),
                             (positional, "three_positional_values")):
            with self.subTest(schema=schema):
                parsed = parse_candidate(text, request())
                self.assertEqual((parsed["answer"], parsed["response_schema"]), ("Paris", schema))
                self.assertFalse(complete_labeled_output(text, CANDIDATE_LABELS))
                self.assertTrue(complete_labeled_output(text + "\n", CANDIDATE_LABELS))

    def test_partial_mixtures_multiple_blocks_and_prose_are_not_salvaged(self):
        for text in ("1\nQUOTE: The capital of France is Paris.\nParis",
                     "1\nThe capital of France is Paris.\nANSWER: Paris",
                     "one\nThe capital of France is Paris.\nParis",
                     CANDIDATE + "\n" + CANDIDATE, CANDIDATE + "\nextra", "1\nParis"):
            with self.subTest(text=text), self.assertRaises(FormatError):
                parse_candidate_lines(text)

    def test_actual_r2_schema_failures_recover_but_grounding_failures_still_reject(self):
        path = Path(__file__).parent / "fixtures/answer_repair_v8r2_candidate_cases.json"
        cases = json.loads(path.read_text())["cases"]
        self.assertEqual(len(cases), 5)
        for case in cases:
            with self.subTest(case=case["name"]):
                if case["expected"] == "grounding":
                    with self.assertRaises(GroundingError):
                        parse_candidate(case["text"], real_request(case))
                else:
                    parsed = parse_candidate(case["text"], real_request(case))
                    self.assertEqual(parsed["status"], case["expected"])
                    self.assertEqual(parsed["response_schema"], case["response_schema"])
                    citation = parsed["citations"][0]
                    source = next(e["text"] for e in case["runtime"]["evidence"] if e["rank"] == citation["rank"])
                    a, b = parsed["source_offsets"]
                    self.assertEqual(citation["quote"], source[a:b])

    def test_rejection_taxonomy_and_raw_replay_do_not_change_fallback(self):
        for bad, kind in (("unstructured", "format"),
                          (CANDIDATE.replace("Paris.", "London."), "grounding")):
            result = repair_answer(request(), SequenceGenerator([WRONG, bad]), strategy="conservative_repair")
            self.assertEqual((result.final_answer, result.status), ("London", "candidate_invalid"))
            self.assertEqual(result.repair_trace["stages"][-1]["validation_kind"], kind)
            self.assertEqual(asdict(repair_from_v8_trace(request(), result.repair_trace)), asdict(result))
        bare = CANDIDATE.removeprefix("RANK: ")
        result = repair_answer(request(), SequenceGenerator([WRONG, bare, check()]), strategy="conservative_repair")
        self.assertEqual((result.final_answer, result.extra_calls), ("Paris", 3))
        self.assertEqual(asdict(repair_from_v8_trace(request(), result.repair_trace)), asdict(result))


if __name__ == "__main__":
    unittest.main()
