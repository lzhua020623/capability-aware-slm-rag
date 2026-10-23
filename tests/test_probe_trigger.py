"""Tests for the prototype Probe trigger."""

from __future__ import annotations

import inspect
import unittest

from src.diagnosis.labels import DiagnosisLabel
from src.probe.trigger import HIGH_THRESHOLD, LOW_THRESHOLD, STAGE2_THRESHOLD, should_probe

CONFIDENT_STAGE2 = {
    DiagnosisLabel.EVIDENCE_INSUFFICIENCY: 0.80,
    DiagnosisLabel.EVIDENCE_ACCESSIBILITY: 0.15,
    DiagnosisLabel.EVIDENCE_UTILISATION: 0.05,
}
UNSURE_STAGE2 = {
    DiagnosisLabel.EVIDENCE_INSUFFICIENCY: 0.45,
    DiagnosisLabel.EVIDENCE_ACCESSIBILITY: 0.35,
    DiagnosisLabel.EVIDENCE_UTILISATION: 0.20,
}


def stage2_with_max(value: float) -> dict:
    rest = (1.0 - value) / 2
    return {
        DiagnosisLabel.EVIDENCE_INSUFFICIENCY: value,
        DiagnosisLabel.EVIDENCE_ACCESSIBILITY: rest,
        DiagnosisLabel.EVIDENCE_UTILISATION: rest,
    }


class TriggerTests(unittest.TestCase):
    def test_prototype_constants(self):
        self.assertEqual((LOW_THRESHOLD, HIGH_THRESHOLD, STAGE2_THRESHOLD), (0.40, 0.55, 0.50))

    def test_band_boundaries_inclusive(self):
        self.assertTrue(should_probe(0.40, CONFIDENT_STAGE2))
        self.assertTrue(should_probe(0.55, CONFIDENT_STAGE2))
        self.assertTrue(should_probe(0.47, None))
        self.assertFalse(should_probe(0.3999, None))
        self.assertFalse(should_probe(0.5501, CONFIDENT_STAGE2))

    def test_stage2_threshold(self):
        self.assertTrue(should_probe(0.80, UNSURE_STAGE2))
        self.assertTrue(should_probe(0.80, stage2_with_max(0.4999)))
        self.assertFalse(should_probe(0.80, stage2_with_max(0.50)))
        self.assertFalse(should_probe(0.80, stage2_with_max(0.51)))

    def test_stage1_correct_not_triggered(self):
        for p in (0.0, 0.10, 0.30, 0.39):
            self.assertFalse(should_probe(p, None))
            self.assertFalse(should_probe(p, UNSURE_STAGE2))

    def test_confident_failure_not_triggered(self):
        for p in (0.56, 0.75, 0.99, 1.0):
            self.assertFalse(should_probe(p, CONFIDENT_STAGE2))

    def test_invalid_inputs(self):
        for bad in (-0.1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                should_probe(bad, CONFIDENT_STAGE2)
        with self.assertRaises(TypeError):
            should_probe("0.5", None)
        with self.assertRaises(ValueError):
            should_probe(0.9, None)
        with self.assertRaises(ValueError):
            should_probe(0.9, {DiagnosisLabel.EVIDENCE_UTILISATION: 1.5})

    def test_trigger_reads_no_offline_data(self):
        params = set(inspect.signature(should_probe).parameters)
        self.assertEqual(params, {"stage1_failure_probability", "stage2_class_scores"})


if __name__ == "__main__":
    unittest.main()
