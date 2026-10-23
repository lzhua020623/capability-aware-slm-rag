"""Tests for annotation reading, normalization mapping and the development join. No models are loaded."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts import normalize_diagnosis_annotations as normalizer
from src.diagnosis.annotations import ANNOTATION_FIELDS, AnnotationRecord, load_annotations, parse_annotation
from src.diagnosis.dev_dataset import join_development_records
from src.diagnosis.labels import DiagnosisLabel
from src.diagnosis.schema import FORBIDDEN_RUNTIME_FIELDS
from tests.test_diagnosis_adapter import fever_record, nq_record


def annotation(sample_id: str, dataset: str = "nq", **overrides) -> dict:
    record = {
        "sample_id": sample_id,
        "dataset": dataset,
        "primary_label": "evidence_accessibility",
        "secondary_label": None,
        "needs_manual_review": False,
        "review_reason": None,
        "annotation_source": "NQ_C1_3216_initial_labels.xlsx",
    }
    return record | overrides


class AnnotationReaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def write(self, rows: list[dict]) -> Path:
        path = Path(self.temp.name) / "labels.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return path

    def test_load_valid_annotations(self):
        rows = [
            annotation("-7299076255108845857"),
            annotation(
                "106023",
                "fever",
                primary_label="evidence_utilisation",
                secondary_label="oracle_recovered_7b",
                needs_manual_review=True,
                review_reason="gold_conflict",
            ),
        ]
        loaded = load_annotations(self.write(rows))
        self.assertEqual(list(loaded), ["-7299076255108845857", "106023"])
        self.assertIs(loaded["106023"].primary_label, DiagnosisLabel.EVIDENCE_UTILISATION)
        self.assertEqual(loaded["106023"].secondary_label, "oracle_recovered_7b")

    def test_duplicate_annotation_id_rejected(self):
        with self.assertRaises(ValueError):
            load_annotations(self.write([annotation("1"), annotation("1")]))

    def test_invalid_primary_label_rejected(self):
        for label in ("Evidence Accessibility Failure", "unresolved_hard_case", ""):
            with self.assertRaises(ValueError):
                load_annotations(self.write([annotation("1", primary_label=label)]))

    def test_gold_field_in_annotation_rejected(self):
        with self.assertRaises(ValueError):
            parse_annotation(annotation("1") | {"gold_answer": "x"})

    def test_secondary_label_only_for_utilisation(self):
        with self.assertRaises(ValueError):
            parse_annotation(annotation("1", secondary_label="oracle_recovered_7b"))

    def test_review_reason_requires_review_flag(self):
        with self.assertRaises(ValueError):
            parse_annotation(annotation("1", review_reason="gold_conflict"))
        with self.assertRaises(ValueError):
            parse_annotation(annotation("1", needs_manual_review=True))

    def test_annotation_fields_have_no_gold(self):
        self.assertFalse(set(ANNOTATION_FIELDS) & FORBIDDEN_RUNTIME_FIELDS)


class NormalizerMappingTests(unittest.TestCase):
    def test_primary_label_mapping(self):
        self.assertEqual(
            {normalizer.map_primary_label(name) for name in normalizer.PRIMARY_LABEL_MAP},
            set(DiagnosisLabel),
        )
        with self.assertRaises(ValueError):
            normalizer.map_primary_label("Unresolved / General Hard Case")

    def test_secondary_label_mapping(self):
        self.assertIsNone(normalizer.map_secondary_label(None))
        self.assertIsNone(normalizer.map_secondary_label(""))
        self.assertEqual(normalizer.map_secondary_label("7B Oracle Recovered"), "oracle_recovered_7b")
        with self.assertRaises(ValueError):
            normalizer.map_secondary_label("Correct / No Failure")

    def test_review_reason_mapping(self):
        self.assertEqual(normalizer.map_review_reason("gold 与提供的证据冲突"), "gold_conflict")
        with self.assertRaises(ValueError):
            normalizer.map_review_reason("unknown review type")

    def test_excel_sample_id(self):
        self.assertEqual(normalizer.excel_sample_id(106023.0), "106023")
        self.assertEqual(normalizer.excel_sample_id(" -7299076255108845857 "), "-7299076255108845857")


class DevelopmentJoinTests(unittest.TestCase):
    def annotations(self, *rows: dict) -> dict[str, AnnotationRecord]:
        return {row["sample_id"]: parse_annotation(row) for row in rows}

    def test_strict_one_to_one_join(self):
        c1 = [nq_record(), fever_record()]
        labels = self.annotations(
            annotation("-7299076255108845857"),
            annotation("106023", "fever", primary_label="correct", needs_manual_review=True, review_reason="ai_flagged"),
        )
        records = join_development_records(c1, labels)
        self.assertEqual([r.input.sample_id for r in records], ["-7299076255108845857", "106023"])
        self.assertIs(records[0].target_label, DiagnosisLabel.EVIDENCE_ACCESSIBILITY)
        self.assertTrue(records[1].metadata.needs_manual_review)

        output = records[1].to_dict()
        self.assertEqual(set(output), {"input", "target_label", "metadata"})
        self.assertEqual(
            set(output["input"]), {"sample_id", "dataset", "query", "evidence", "initial_answer"}
        )
        self.assertEqual(output["input"]["dataset"], "fever")
        self.assertEqual(output["input"]["evidence"][0], {
            "rank": 1, "passage_id": "Boeing_787_Dreamliner", "text": "The Boeing 787 Dreamliner is ...", "score": 0.5653,
        })
        self.assertEqual(output["target_label"], "correct")
        serialized = json.dumps(output)
        for forbidden in ("gold", "\"correct\":", "\"em\"", "\"f1\"", "raw_output", "latency"):
            self.assertNotIn(forbidden, serialized)

    def test_missing_annotation_rejected(self):
        with self.assertRaises(ValueError):
            join_development_records([nq_record(), fever_record()], self.annotations(annotation("-7299076255108845857")))

    def test_extra_annotation_rejected(self):
        labels = self.annotations(annotation("-7299076255108845857"), annotation("999"))
        with self.assertRaises(ValueError):
            join_development_records([nq_record()], labels)

    def test_duplicate_c1_record_rejected(self):
        with self.assertRaises(ValueError):
            join_development_records([nq_record(), nq_record()], self.annotations(annotation("-7299076255108845857")))

    def test_dataset_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            join_development_records([fever_record()], self.annotations(annotation("106023", "nq")))


if __name__ == "__main__":
    unittest.main()
