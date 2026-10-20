"""Tests for the NQ_15 annotation normalizer. No workbook and no model is read."""

from __future__ import annotations

import json
import unittest

from scripts import normalize_nq15_annotations as normalizer
from src.diagnosis.labels import DiagnosisLabel
from src.diagnosis.schema import FORBIDDEN_RUNTIME_FIELDS

EVIDENCE = [
    {"rank": 1, "passage_id": "nq_p_1", "text": "first passage", "score": 0.9},
    {"rank": 2, "passage_id": "nq_p_2", "text": "second passage", "score": 0.8},
]


class PrimaryLabelTest(unittest.TestCase):
    def test_every_returned_spelling_maps_to_the_frozen_taxonomy(self):
        cases = {
            "Correct": DiagnosisLabel.CORRECT,
            "Correct / No Failure": DiagnosisLabel.CORRECT,
            "Evidence Insufficiency": DiagnosisLabel.EVIDENCE_INSUFFICIENCY,
            "Evidence Accessibility Failure": DiagnosisLabel.EVIDENCE_ACCESSIBILITY,
            "Evidence Utilisation Failure": DiagnosisLabel.EVIDENCE_UTILISATION,
        }
        for text, label in cases.items():
            self.assertIs(normalizer.parse_primary_label(text), label)
            self.assertIs(normalizer.parse_primary_label(f"  {text}  "), label)

    def test_the_map_covers_the_whole_taxonomy_and_nothing_else(self):
        self.assertEqual(set(normalizer.PRIMARY_LABEL_MAP.values()), set(DiagnosisLabel))

    def test_unknown_or_missing_labels_are_rejected(self):
        for value in ("Accessibility", "accessibility", "", "   ", None, 3):
            with self.assertRaises(ValueError):
                normalizer.parse_primary_label(value)


class EvidenceTest(unittest.TestCase):
    def test_parses_json_text_and_coerces_types(self):
        raw = json.dumps([{"rank": "1", "passage_id": 7, "text": "a", "score": "0.5"}])
        parsed = normalizer.parse_evidence(raw)
        self.assertEqual(parsed, [{"rank": 1, "passage_id": "7", "text": "a", "score": 0.5}])

    def test_accepts_an_already_parsed_list(self):
        self.assertEqual(normalizer.parse_evidence(EVIDENCE), EVIDENCE)

    def test_ranks_must_be_one_to_n(self):
        for ranks in ([2, 3], [1, 1], [1, 3]):
            broken = [dict(passage, rank=rank) for passage, rank in zip(EVIDENCE, ranks)]
            with self.assertRaises(ValueError):
                normalizer.parse_evidence(broken)

    def test_empty_or_malformed_evidence_is_rejected(self):
        for value in ("", "[]", "not json", json.dumps([1, 2])):
            with self.assertRaises((ValueError, json.JSONDecodeError)):
                normalizer.parse_evidence(value)


class ReferenceTest(unittest.TestCase):
    def test_keeps_only_the_declared_offline_fields(self):
        raw = json.dumps(
            {
                "gold_answer": "Paris",
                "gold_passage_id": "nq_p_1",
                "gold_in_top5": True,
                "em": 1,
                "correct": True,
                "secret": "drop me",
            }
        )
        parsed = normalizer.parse_reference(raw)
        self.assertEqual(set(parsed), set(normalizer.REFERENCE_FIELDS))
        self.assertEqual(parsed["gold_answer"], "Paris")
        self.assertNotIn("secret", parsed)

    def test_missing_fields_become_none(self):
        self.assertIsNone(normalizer.parse_reference("{}")["gold_answer"])

    def test_only_the_declared_gold_references_survive(self):
        # The three gold_* fields are the offline supervision the annotation guide allows here.
        # Everything else runtime-forbidden (em, f1, correct, c0/c3, larger-model outcomes) must be dropped.
        declared_gold = {"gold_answer", "gold_evidence_text", "gold_passage_id"}
        parsed = normalizer.parse_reference(json.dumps({field: 1 for field in FORBIDDEN_RUNTIME_FIELDS}))
        self.assertEqual((set(parsed) & FORBIDDEN_RUNTIME_FIELDS) - declared_gold, set())
        self.assertNotIn("gold_evidence_in_top5", parsed)

    def test_non_object_reference_is_rejected(self):
        with self.assertRaises(ValueError):
            normalizer.parse_reference(json.dumps([1, 2]))


if __name__ == "__main__":
    unittest.main()
