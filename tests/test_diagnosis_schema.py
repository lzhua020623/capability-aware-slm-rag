"""Interface tests for the frozen Diagnosis labels and runtime schema. No models are loaded."""

from __future__ import annotations

import dataclasses
import unittest

from src.diagnosis import DatasetName, DiagnosisInput, DiagnosisLabel, DiagnosisOutput, EvidenceItem
from src.diagnosis.labels import RUNTIME_LABELS
from src.diagnosis.schema import FORBIDDEN_RUNTIME_FIELDS


def make_evidence() -> list[EvidenceItem]:
    return [
        EvidenceItem(rank=1, passage_id="Boeing_787_Dreamliner", text="The Boeing 787 ...", score=0.5653),
        EvidenceItem(rank=2, passage_id="Boeing_707", text="The Boeing 707 ...", score=0.4211),
    ]


class DiagnosisLabelTests(unittest.TestCase):
    def test_four_runtime_labels(self):
        self.assertEqual(
            [label.value for label in RUNTIME_LABELS],
            [
                "correct",
                "evidence_insufficiency",
                "evidence_accessibility",
                "evidence_utilisation",
            ],
        )
        self.assertIs(DiagnosisLabel("evidence_utilisation"), DiagnosisLabel.EVIDENCE_UTILISATION)
        self.assertEqual(DiagnosisLabel.CORRECT, "correct")

    def test_offline_categories_are_not_runtime_labels(self):
        values = {label.value for label in RUNTIME_LABELS}
        for offline in (
            "oracle_recovered_7b",
            "likely_small_model_capability_limitation",
            "unresolved_hard_case",
        ):
            self.assertNotIn(offline, values)


class DiagnosisInputTests(unittest.TestCase):
    def test_build_input(self):
        item = DiagnosisInput(
            sample_id="-7299076255108845857",
            dataset="nq",
            query="what is the sentencing reform act of 1984",
            evidence=make_evidence(),
            initial_answer="U.S. Sentencing Commission",
        )
        self.assertEqual(item.sample_id, "-7299076255108845857")
        self.assertIs(item.dataset, DatasetName.NQ)
        self.assertEqual(len(item.evidence), 2)
        self.assertEqual(item.initial_answer, "U.S. Sentencing Commission")

    def test_fever_integer_sample_id_becomes_string(self):
        item = DiagnosisInput(
            sample_id=106023,
            dataset="fever",
            query="Boeing 787 Dreamliner is a machine.",
            evidence=make_evidence(),
            initial_answer="SUPPORTS",
        )
        self.assertIsInstance(item.sample_id, str)
        self.assertEqual(item.sample_id, "106023")

    def test_canonical_datasets_accepted(self):
        for value, expected in (("nq", DatasetName.NQ), ("fever", DatasetName.FEVER), (DatasetName.FEVER, DatasetName.FEVER)):
            item = DiagnosisInput(sample_id="1", dataset=value, query="q", evidence=[], initial_answer="a")
            self.assertIs(item.dataset, expected)

    def test_non_canonical_datasets_rejected(self):
        for value in ("natural_questions", "NQ", "FEVER", "hotpotqa", "", None):
            with self.assertRaises(ValueError):
                DiagnosisInput(sample_id="1", dataset=value, query="q", evidence=[], initial_answer="a")

    def test_evidence_fields_are_preserved(self):
        item = DiagnosisInput(
            sample_id="1",
            dataset="fever",
            query="claim",
            evidence=make_evidence(),
            initial_answer="SUPPORTS",
        )
        first, second = item.evidence
        self.assertEqual((first.rank, first.passage_id, first.score), (1, "Boeing_787_Dreamliner", 0.5653))
        self.assertEqual((second.rank, second.passage_id, second.score), (2, "Boeing_707", 0.4211))
        self.assertIsInstance(first.score, float)

    def test_duplicate_evidence_rank_rejected(self):
        evidence = make_evidence()
        evidence.append(EvidenceItem(rank=1, passage_id="dup", text="", score=0.1))
        with self.assertRaises(ValueError):
            DiagnosisInput(sample_id="1", dataset="nq", query="q", evidence=evidence, initial_answer="a")

    def test_invalid_evidence_rank_rejected(self):
        with self.assertRaises(ValueError):
            EvidenceItem(rank=0, passage_id="p", text="t", score=0.5)


class DiagnosisOutputTests(unittest.TestCase):
    def test_valid_output(self):
        output = DiagnosisOutput(diagnosis_label="evidence_accessibility", confidence=0.7, needs_probe=False)
        self.assertIs(output.diagnosis_label, DiagnosisLabel.EVIDENCE_ACCESSIBILITY)
        self.assertEqual(output.confidence, 0.7)
        self.assertIsNone(output.class_scores)

    def test_confidence_below_zero_rejected(self):
        with self.assertRaises(ValueError):
            DiagnosisOutput(diagnosis_label=DiagnosisLabel.CORRECT, confidence=-0.01, needs_probe=False)

    def test_confidence_above_one_rejected(self):
        with self.assertRaises(ValueError):
            DiagnosisOutput(diagnosis_label=DiagnosisLabel.CORRECT, confidence=1.01, needs_probe=False)

    def test_invalid_label_rejected(self):
        for label in ("unresolved_hard_case", "Evidence Utilisation Failure", ""):
            with self.assertRaises(ValueError):
                DiagnosisOutput(diagnosis_label=label, confidence=0.5, needs_probe=True)

    def test_needs_probe_must_be_bool(self):
        with self.assertRaises(TypeError):
            DiagnosisOutput(diagnosis_label=DiagnosisLabel.CORRECT, confidence=0.5, needs_probe=1)

    def test_optional_class_scores(self):
        output = DiagnosisOutput(
            diagnosis_label=DiagnosisLabel.CORRECT,
            confidence=0.6,
            needs_probe=True,
            class_scores={"correct": 0.6, DiagnosisLabel.EVIDENCE_UTILISATION: 0.4},
        )
        self.assertEqual(
            output.class_scores,
            {DiagnosisLabel.CORRECT: 0.6, DiagnosisLabel.EVIDENCE_UTILISATION: 0.4},
        )
        with self.assertRaises(ValueError):
            DiagnosisOutput(
                diagnosis_label=DiagnosisLabel.CORRECT,
                confidence=0.6,
                needs_probe=True,
                class_scores={"unresolved_hard_case": 0.4},
            )


class RuntimeSchemaLeakageTests(unittest.TestCase):
    def test_no_gold_or_outcome_fields(self):
        for schema in (EvidenceItem, DiagnosisInput, DiagnosisOutput):
            names = {f.name for f in dataclasses.fields(schema)}
            self.assertFalse(names & FORBIDDEN_RUNTIME_FIELDS, f"{schema.__name__} leaks {names & FORBIDDEN_RUNTIME_FIELDS}")
            for name in names:
                self.assertNotIn("gold", name)

    def test_input_fields_are_frozen(self):
        self.assertEqual(
            [f.name for f in dataclasses.fields(DiagnosisInput)],
            ["sample_id", "dataset", "query", "evidence", "initial_answer"],
        )
        self.assertEqual(
            [f.name for f in dataclasses.fields(EvidenceItem)],
            ["rank", "passage_id", "text", "score"],
        )


if __name__ == "__main__":
    unittest.main()
