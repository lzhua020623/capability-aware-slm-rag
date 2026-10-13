"""Tests for the Evidence Probe schema. No model is loaded."""

from __future__ import annotations

import dataclasses
import unittest

from src.diagnosis.labels import DiagnosisLabel
from src.diagnosis.schema import FORBIDDEN_RUNTIME_FIELDS
from src.probe.schema import (
    AnswerSupport,
    EvidenceAccessibility,
    EvidenceSufficiency,
    ProbeLabel,
    ProbeResult,
    parse_ranks,
)


def result(**overrides) -> ProbeResult:
    values = dict(
        answer_supported="unsupported",
        evidence_sufficiency="sufficient",
        evidence_accessibility="clear",
        supporting_ranks=(),
        conflicting_ranks=(2,),
    )
    values.update(overrides)
    return ProbeResult.from_judgments(**values)


class ProbeSchemaTests(unittest.TestCase):
    def test_enum_values(self):
        self.assertEqual({m.value for m in AnswerSupport}, {"supported", "unsupported", "uncertain"})
        self.assertEqual({m.value for m in EvidenceSufficiency}, {"sufficient", "insufficient", "uncertain"})
        self.assertEqual({m.value for m in EvidenceAccessibility}, {"clear", "hard_to_access", "uncertain"})
        self.assertEqual(
            {m.value for m in ProbeLabel},
            {label.value for label in DiagnosisLabel} | {"uncertain"},
        )

    def test_illegal_enum_rejected(self):
        for field, bad in (
            ("answer_supported", "partially"),
            ("answer_supported", "SUPPORTED"),
            ("evidence_sufficiency", "enough"),
            ("evidence_accessibility", "hard"),
            ("evidence_accessibility", None),
        ):
            with self.subTest(field=field, value=bad), self.assertRaises(ValueError):
                result(**{field: bad})

    def test_rank_bounds(self):
        self.assertEqual(result(supporting_ranks=[3, 1]).supporting_ranks, (1, 3))
        for bad in ([0], [6], [1, 1], [True], ["1"], [1.0]):
            with self.subTest(ranks=bad), self.assertRaises(ValueError):
                result(supporting_ranks=bad)
            with self.subTest(conflicting=bad), self.assertRaises(ValueError):
                result(conflicting_ranks=bad)
        with self.assertRaises(ValueError):
            parse_ranks("supporting_ranks", [4], max_rank=3)
        with self.assertRaises(ValueError):
            parse_ranks("supporting_ranks", "12")

    def test_probe_label_must_follow_mapping(self):
        valid = result()
        self.assertIs(valid.probe_label, ProbeLabel.EVIDENCE_UTILISATION)
        with self.assertRaises(ValueError):
            dataclasses.replace(valid, probe_label=ProbeLabel.CORRECT)
        with self.assertRaises(ValueError):
            dataclasses.replace(valid, probe_label="unknown")

    def test_to_diagnosis_label(self):
        self.assertIsNone(ProbeLabel.UNCERTAIN.to_diagnosis_label())
        for label in DiagnosisLabel:
            self.assertIs(ProbeLabel(label.value).to_diagnosis_label(), label)

    def test_schema_has_no_offline_fields(self):
        fields = {f.name for f in dataclasses.fields(ProbeResult)}
        offline = FORBIDDEN_RUNTIME_FIELDS | {"target_label", "metadata", "sample_id", "c3", "oracle", "label"}
        self.assertFalse(fields & offline)
        self.assertFalse({name for name in fields if any(word in name for word in ("gold", "target", "c3", "oracle"))})

    def test_unparsable_is_all_uncertain(self):
        fallback = ProbeResult.unparsable("not json", "No JSON object found")
        self.assertIs(fallback.probe_label, ProbeLabel.UNCERTAIN)
        self.assertEqual(fallback.parse_error, "No JSON object found")
        self.assertEqual(fallback.to_dict()["probe_label"], "uncertain")


if __name__ == "__main__":
    unittest.main()
