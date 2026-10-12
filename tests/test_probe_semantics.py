"""Regression tests for the FEVER claim/verdict inversion and NQ constraint checks. No model is loaded."""

from __future__ import annotations

import itertools
import json
import unittest

from src.diagnosis.schema import DiagnosisInput, EvidenceItem
from src.probe.probe import EvidenceProbe, ProbeParseError, parse_fever_probe_output
from src.probe.prompts import build_probe_prompt
from src.probe.schema import (
    AnswerSupport,
    EvidenceAccessibility,
    EvidenceSufficiency,
    FeverVerdict,
    ProbeLabel,
    fever_judgments,
)
from tests.test_probe_mapping import RecordingGenerator


def brubaker(initial_answer: str) -> DiagnosisInput:
    return DiagnosisInput(
        sample_id="222046",
        dataset="fever",
        query="Brubaker is a 2007 prison drama.",
        evidence=[
            EvidenceItem(rank=1, passage_id="Brubaker", text="Brubaker is a 1980 American prison drama film .", score=0.7),
            EvidenceItem(rank=2, passage_id="Robert_Redford", text="Robert Redford is an American actor .", score=0.5),
        ],
        initial_answer=initial_answer,
    )


def party_song() -> DiagnosisInput:
    return DiagnosisInput(
        sample_id="9148",
        dataset="nq",
        query="who sang it my party and i'll cry if i want to in the 80",
        evidence=[
            EvidenceItem(rank=1, passage_id="p1", text="It's My Party is a 1963 hit song by Lesley Gore .", score=0.8),
        ],
        initial_answer="Lesley Gore",
    )


def fever_json(verdict: str, accessibility: str = "clear", decisive=(1,), opposing=()) -> str:
    return json.dumps({
        "evidence_verdict": verdict,
        "evidence_accessibility": accessibility,
        "decisive_ranks": list(decisive),
        "opposing_ranks": list(opposing),
        "reason": "Passage 1 dates the film to 1980.",
    })


class FeverVerdictTests(unittest.TestCase):
    def test_case1_refutes_prediction_backed_by_contradicting_evidence(self):
        result = EvidenceProbe(RecordingGenerator(fever_json("REFUTES"))).run(brubaker("REFUTES"))
        self.assertIs(result.evidence_verdict, FeverVerdict.REFUTES)
        self.assertIs(result.answer_supported, AnswerSupport.SUPPORTED)
        self.assertIs(result.evidence_sufficiency, EvidenceSufficiency.SUFFICIENT)
        self.assertIs(result.probe_label, ProbeLabel.CORRECT)
        self.assertEqual((result.supporting_ranks, result.conflicting_ranks), ((1,), ()))

    def test_case2_supports_prediction_on_refuted_claim(self):
        result = EvidenceProbe(RecordingGenerator(fever_json("REFUTES"))).run(brubaker("SUPPORTS"))
        self.assertIs(result.evidence_verdict, FeverVerdict.REFUTES)
        self.assertIs(result.answer_supported, AnswerSupport.UNSUPPORTED)
        self.assertIs(result.probe_label, ProbeLabel.EVIDENCE_UTILISATION)
        self.assertEqual((result.supporting_ranks, result.conflicting_ranks), ((), (1,)))
        hard = EvidenceProbe(RecordingGenerator(fever_json("REFUTES", "hard_to_access"))).run(brubaker("SUPPORTS"))
        self.assertIs(hard.probe_label, ProbeLabel.EVIDENCE_ACCESSIBILITY)

    def test_case3_insufficient_evidence(self):
        for initial in ("SUPPORTS", "REFUTES"):
            result = EvidenceProbe(RecordingGenerator(fever_json("INSUFFICIENT", "clear"))).run(brubaker(initial))
            self.assertIs(result.evidence_verdict, FeverVerdict.INSUFFICIENT)
            self.assertIs(result.answer_supported, AnswerSupport.UNSUPPORTED)
            self.assertIs(result.evidence_sufficiency, EvidenceSufficiency.INSUFFICIENT)
            self.assertIs(result.evidence_accessibility, EvidenceAccessibility.UNCERTAIN)
            self.assertIs(result.probe_label, ProbeLabel.EVIDENCE_INSUFFICIENCY)

    def test_comparison_table(self):
        for verdict, initial, access in itertools.product(FeverVerdict, ("SUPPORTS", "REFUTES"), EvidenceAccessibility):
            support, sufficiency, accessibility = fever_judgments(verdict, initial, access)
            if verdict is FeverVerdict.INSUFFICIENT:
                expected = (AnswerSupport.UNSUPPORTED, EvidenceSufficiency.INSUFFICIENT, EvidenceAccessibility.UNCERTAIN)
            else:
                expected = (
                    AnswerSupport.SUPPORTED if verdict.value == initial else AnswerSupport.UNSUPPORTED,
                    EvidenceSufficiency.SUFFICIENT,
                    access,
                )
            self.assertEqual((support, sufficiency, accessibility), expected)
        self.assertIs(fever_judgments("REFUTES", " refutes ", "clear")[0], AnswerSupport.SUPPORTED)

    def test_invalid_fever_outputs(self):
        for text in (
            fever_json("NOT ENOUGH INFO"),
            fever_json("supports"),
            fever_json("REFUTES", "obvious"),
            fever_json("REFUTES", decisive=(6,)),
            json.dumps({"answer_supported": "supported", "evidence_sufficiency": "sufficient",
                        "evidence_accessibility": "clear", "supporting_ranks": [1], "conflicting_ranks": []}),
        ):
            with self.subTest(text=text), self.assertRaises(ProbeParseError):
                parse_fever_probe_output(text, "REFUTES")
        fallback = EvidenceProbe(RecordingGenerator(fever_json("MAYBE"))).run(brubaker("REFUTES"))
        self.assertIs(fallback.probe_label, ProbeLabel.UNCERTAIN)

    def test_fever_prompt_hides_initial_answer(self):
        refutes_prompt = build_probe_prompt(brubaker("REFUTES"))
        self.assertEqual(refutes_prompt, build_probe_prompt(brubaker("SUPPORTS")))
        self.assertIn("Claim: Brubaker is a 2007 prison drama.", refutes_prompt)
        self.assertIn("evidence_verdict", refutes_prompt)
        self.assertIn("INSUFFICIENT", refutes_prompt)
        for absent in ("Predicted verdict", "initial answer", "proposed answer", "answer_supported"):
            self.assertNotIn(absent.lower(), refutes_prompt.lower())


class NqConstraintPromptTests(unittest.TestCase):
    def test_prompt_requires_exact_question_constraints(self):
        prompt = build_probe_prompt(party_song())
        self.assertIn("Semantic relevance alone is not enough.", prompt)
        self.assertIn("as an answer to the exact question", prompt)
        self.assertIn("Check all temporal, entity, version, quantity and scope constraints", prompt)
        self.assertIn("1980s version", prompt)
        self.assertIn("1963 version), the answer is unsupported, even if the topic matches", prompt)
        self.assertIn("enough information to answer the exact question", prompt)
        self.assertIn('"insufficient" if a key fact or constraint is missing', prompt)
        self.assertIn("Question: who sang it my party and i'll cry if i want to in the 80", prompt)
        self.assertIn("Proposed answer: Lesley Gore", prompt)

    def test_unsupported_version_mismatch_maps_to_failure(self):
        output = json.dumps({
            "answer_supported": "unsupported", "evidence_sufficiency": "insufficient",
            "evidence_accessibility": "uncertain", "supporting_ranks": [], "conflicting_ranks": [],
            "reason": "Passage 1 covers only the 1963 version.",
        })
        result = EvidenceProbe(RecordingGenerator(output)).run(party_song())
        self.assertIs(result.answer_supported, AnswerSupport.UNSUPPORTED)
        self.assertIs(result.probe_label, ProbeLabel.EVIDENCE_INSUFFICIENCY)
        self.assertIsNone(result.evidence_verdict)


if __name__ == "__main__":
    unittest.main()
