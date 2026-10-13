"""Tests for the hierarchical Diagnosis classifier. Synthetic inputs only; no LLM is loaded."""

from __future__ import annotations

import dataclasses
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from src.diagnosis.classifier import LABEL_ORDER, feature_matrix, feature_names_for
from src.diagnosis.features import FEATURE_NAMES
from src.diagnosis.hierarchical_classifier import (
    FAILURE_LABELS,
    HierarchicalDiagnosisClassifier,
)
from src.diagnosis.labels import DiagnosisLabel
from src.diagnosis.schema import FORBIDDEN_RUNTIME_FIELDS, DiagnosisOutput
from tests.test_diagnosis_classifier import synthetic_split

ROOT = Path(__file__).resolve().parents[1]


def with_threshold(classifier: HierarchicalDiagnosisClassifier, threshold: float) -> HierarchicalDiagnosisClassifier:
    return dataclasses.replace(classifier, stage1_threshold=threshold)


class HierarchicalClassifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train_inputs, cls.train_labels = synthetic_split(80, seed=1)
        cls.other_inputs, _ = synthetic_split(20, seed=2)
        cls.classifier = HierarchicalDiagnosisClassifier.fit(
            cls.train_inputs, cls.train_labels, stage1_c=1.0, stage1_class_weight=None, stage2_c=1.0,
            stage1_threshold=0.5,
        )
        cls.matrix = feature_matrix(cls.other_inputs, cls.classifier.feature_names)

    def test_stage1_failure_probability(self):
        p = self.classifier.stage1_failure_probability(self.matrix)
        self.assertEqual(p.shape, (len(self.other_inputs),))
        self.assertTrue(np.all((p >= 0.0) & (p <= 1.0)))
        diagnoses = self.classifier.diagnose_matrix(self.matrix)
        np.testing.assert_allclose([d.stage1_failure_probability for d in diagnoses], p)

    def test_stage1_threshold_is_applied(self):
        p = self.classifier.stage1_failure_probability(self.matrix)
        threshold = float(np.median(p))
        diagnoses = with_threshold(self.classifier, threshold).diagnose_matrix(self.matrix)
        self.assertEqual([d.stage1_label == "failure" for d in diagnoses], (p >= threshold).tolist())
        self.assertTrue(all(d.stage1_label == "failure" for d in with_threshold(self.classifier, 0.0).diagnose_matrix(self.matrix)))

    def test_stage2_not_called_when_stage1_says_correct(self):
        classifier = with_threshold(self.classifier, 1.0)
        with mock.patch.object(classifier.stage2, "predict_proba", side_effect=AssertionError("Stage 2 called")):
            diagnoses = classifier.diagnose_matrix(self.matrix)
        for d in diagnoses:
            self.assertEqual(d.stage1_label, "correct")
            self.assertIs(d.final_label, DiagnosisLabel.CORRECT)
            self.assertIsNone(d.stage2_class_scores)
            self.assertAlmostEqual(d.confidence, 1.0 - d.stage1_failure_probability)

    def test_stage2_called_when_stage1_says_failure(self):
        classifier = with_threshold(self.classifier, 0.0)
        with mock.patch.object(classifier.stage2, "predict_proba", wraps=classifier.stage2.predict_proba) as spy:
            diagnoses = classifier.diagnose_matrix(self.matrix)
        self.assertEqual(spy.call_count, 1)
        self.assertEqual(spy.call_args.args[0].shape[0], len(self.other_inputs))
        for d in diagnoses:
            self.assertIn(d.final_label, FAILURE_LABELS)
            self.assertEqual(set(d.stage2_class_scores), set(FAILURE_LABELS))
            self.assertIs(d.final_label, max(d.stage2_class_scores, key=d.stage2_class_scores.get))

    def test_stage2_has_only_failure_labels(self):
        self.assertEqual(self.classifier.stage2.labels, tuple(label.value for label in FAILURE_LABELS))
        self.assertNotIn(DiagnosisLabel.CORRECT, FAILURE_LABELS)
        self.assertEqual(self.classifier.stage2.model.coef_.shape[0], 3)
        n_failures = sum(label is not DiagnosisLabel.CORRECT for label in self.train_labels)
        self.assertEqual(self.classifier.stage2.scaler.n_samples_seen_, n_failures)
        self.assertEqual(self.classifier.stage1.scaler.n_samples_seen_, len(self.train_inputs))

    def test_final_output_uses_runtime_labels(self):
        for threshold in (0.0, 0.5, 1.0):
            classifier = with_threshold(self.classifier, threshold)
            for item in self.other_inputs:
                output = classifier.predict(item)
                self.assertIsInstance(output, DiagnosisOutput)
                self.assertIn(output.diagnosis_label, LABEL_ORDER)
                self.assertFalse(output.needs_probe)

    def test_probability_ranges(self):
        for d in with_threshold(self.classifier, 0.0).diagnose_matrix(self.matrix):
            self.assertTrue(0.0 <= d.confidence <= 1.0)
            self.assertEqual(set(d.class_scores), set(LABEL_ORDER))
            self.assertAlmostEqual(sum(d.class_scores.values()), 1.0)
            self.assertAlmostEqual(sum(d.stage2_class_scores.values()), 1.0)
            self.assertAlmostEqual(d.confidence, d.class_scores[d.final_label])
            self.assertTrue(all(0.0 <= v <= 1.0 for v in d.class_scores.values()))

    def test_probe_threshold_passthrough(self):
        item = self.other_inputs[0]
        self.assertFalse(self.classifier.predict(item).needs_probe)
        self.assertTrue(self.classifier.predict(item, probe_threshold=1.0).needs_probe)
        with self.assertRaises(ValueError):
            self.classifier.predict(item, probe_threshold=1.5)

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "diagnosis" / "hierarchical.json"
            self.classifier.save(path)
            loaded = HierarchicalDiagnosisClassifier.load(path)
        self.assertEqual(loaded.feature_names, self.classifier.feature_names)
        self.assertEqual(loaded.stage1_threshold, self.classifier.stage1_threshold)
        self.assertEqual((loaded.stage1.c, loaded.stage1.class_weight), (1.0, None))
        self.assertEqual((loaded.stage2.c, loaded.stage2.class_weight), (1.0, "balanced"))
        for threshold in (0.0, 0.5):
            original = with_threshold(self.classifier, threshold).diagnose_matrix(self.matrix)
            reloaded = with_threshold(loaded, threshold).diagnose_matrix(self.matrix)
            for a, b in zip(original, reloaded):
                self.assertIs(a.final_label, b.final_label)
                self.assertAlmostEqual(a.confidence, b.confidence, places=12)
                self.assertAlmostEqual(a.stage1_failure_probability, b.stage1_failure_probability, places=12)

    def test_feature_order_is_fixed(self):
        self.assertEqual(self.classifier.feature_names, FEATURE_NAMES)
        with self.assertRaises(ValueError):
            dataclasses.replace(self.classifier, feature_names=tuple(reversed(FEATURE_NAMES)))
        with self.assertRaises(ValueError):
            dataclasses.replace(self.classifier, feature_names=feature_names_for(False))
        with self.assertRaises(ValueError):
            self.classifier.diagnose_matrix(self.matrix[:, :-1])

    def test_no_target_metadata_or_gold(self):
        offline = FORBIDDEN_RUNTIME_FIELDS | {"sample_id", "target_label", "metadata", "needs_manual_review"}
        self.assertFalse(offline & set(self.classifier.feature_names))
        record = {"input": {}, "target_label": "correct", "metadata": {"needs_manual_review": False}}
        with self.assertRaises(TypeError):
            HierarchicalDiagnosisClassifier.fit([record] * 4, list(LABEL_ORDER), 1.0, None, 1.0)
        item = self.other_inputs[0]
        renamed = dataclasses.replace(item, sample_id="different-id")
        self.assertEqual(self.classifier.diagnose(item), self.classifier.diagnose(renamed))

    def test_stage1_threshold_bounds(self):
        for valid in (0.0, 0.3, 1.0):
            self.assertEqual(with_threshold(self.classifier, valid).stage1_threshold, valid)
        for invalid in (-0.01, 1.01, float("nan")):
            with self.assertRaises(ValueError):
                with_threshold(self.classifier, invalid)
        for wrong_type in ("0.5", True, None):
            with self.assertRaises(TypeError):
                with_threshold(self.classifier, wrong_type)

    def test_does_not_load_any_language_model(self):
        code = (
            "import sys; import src.diagnosis.hierarchical_classifier; "
            "loaded = [m for m in ('torch', 'transformers', 'sentence_transformers', 'vllm') if m in sys.modules]; "
            "sys.exit(1 if loaded else 0)"
        )
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
