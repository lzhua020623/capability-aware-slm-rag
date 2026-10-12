"""Tests for the Diagnosis v1 classifier. Synthetic inputs only; no LLM is loaded."""

from __future__ import annotations

import dataclasses
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.diagnosis.classifier import (
    LABEL_ORDER,
    REFUTES_FEATURE,
    DiagnosisClassifier,
    feature_matrix,
    feature_names_for,
)
from src.diagnosis.features import FEATURE_NAMES
from src.diagnosis.labels import DiagnosisLabel
from src.diagnosis.schema import FORBIDDEN_RUNTIME_FIELDS, DiagnosisInput, EvidenceItem

ANSWERS = ("Max Planck", "1900", "Albert Einstein", "not mentioned in the context", "German")
FEVER_ANSWERS = ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO")
PASSAGES = (
    "The quantum theory began in 1900 .",
    "Max Planck proposed the quantum hypothesis .",
    "Planck , Max Planck was German .",
    "Einstein explained the photoelectric effect .",
    "A city with many old buildings .",
)


def synthetic_input(index: int, rng: random.Random) -> DiagnosisInput:
    dataset = "nq" if index % 2 == 0 else "fever"
    answers = ANSWERS if dataset == "nq" else FEVER_ANSWERS
    scores = sorted((rng.uniform(0.2, 0.9) for _ in range(5)), reverse=True)
    passages = rng.sample(PASSAGES, k=5)
    return DiagnosisInput(
        sample_id=str(index),
        dataset=dataset,
        query=rng.choice(("who proposed the quantum hypothesis", "when did quantum theory begin",
                          "Max Planck was German.", "how many photons are emitted")),
        evidence=[
            EvidenceItem(rank=rank, passage_id=f"p{rank}", text=text, score=score)
            for rank, (text, score) in enumerate(zip(passages, scores), start=1)
        ],
        initial_answer=rng.choice(answers),
    )


def synthetic_split(size: int, seed: int) -> tuple[list[DiagnosisInput], list[DiagnosisLabel]]:
    rng = random.Random(seed)
    inputs = [synthetic_input(i, rng) for i in range(size)]
    labels = [LABEL_ORDER[i % len(LABEL_ORDER)] for i in range(size)]
    return inputs, labels


class ClassifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train_inputs, cls.train_labels = synthetic_split(80, seed=1)
        cls.other_inputs, _ = synthetic_split(20, seed=2)
        cls.classifier = DiagnosisClassifier.fit(cls.train_inputs, cls.train_labels, c=1.0)

    def test_scaler_is_fitted_on_train_only(self):
        train_matrix = feature_matrix(self.train_inputs, self.classifier.feature_names)
        mean_before = self.classifier.scaler.mean_.copy()
        np.testing.assert_allclose(mean_before, train_matrix.mean(axis=0))
        self.assertEqual(self.classifier.scaler.n_samples_seen_, len(self.train_inputs))
        self.classifier.predict_proba_batch(self.other_inputs)
        np.testing.assert_array_equal(self.classifier.scaler.mean_, mean_before)
        self.assertEqual(self.classifier.scaler.n_samples_seen_, len(self.train_inputs))

    def test_feature_order_is_fixed(self):
        self.assertEqual(self.classifier.feature_names, FEATURE_NAMES)
        ablated = feature_names_for(include_refutes_prediction=False)
        self.assertNotIn(REFUTES_FEATURE, ablated)
        self.assertEqual(ablated, tuple(name for name in FEATURE_NAMES if name != REFUTES_FEATURE))
        no_refutes = DiagnosisClassifier.fit(self.train_inputs, self.train_labels, c=1.0,
                                             include_refutes_prediction=False)
        self.assertEqual(no_refutes.feature_names, ablated)
        self.assertEqual(no_refutes.model.coef_.shape, (4, len(ablated)))

    def test_outputs_four_probabilities_summing_to_one(self):
        proba = self.classifier.predict_proba_batch(self.other_inputs)
        self.assertEqual(proba.shape, (len(self.other_inputs), 4))
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)

    def test_prediction_contract(self):
        for item in self.other_inputs:
            output = self.classifier.predict(item)
            self.assertEqual(set(output.class_scores), set(LABEL_ORDER))
            best = max(output.class_scores, key=output.class_scores.get)
            self.assertIs(output.diagnosis_label, best)
            self.assertAlmostEqual(output.confidence, output.class_scores[best])
            self.assertTrue(0.0 <= output.confidence <= 1.0)
            self.assertAlmostEqual(sum(output.class_scores.values()), 1.0)

    def test_needs_probe_false_without_threshold(self):
        for item in self.other_inputs:
            self.assertFalse(self.classifier.predict(item).needs_probe)
            self.assertFalse(self.classifier.predict(item, probe_threshold=None).needs_probe)

    def test_probe_threshold_logic(self):
        item = self.other_inputs[0]
        confidence = self.classifier.predict(item).confidence
        self.assertTrue(self.classifier.predict(item, probe_threshold=1.0).needs_probe)
        self.assertFalse(self.classifier.predict(item, probe_threshold=0.0).needs_probe)
        self.assertFalse(self.classifier.predict(item, probe_threshold=confidence).needs_probe)
        self.assertTrue(self.classifier.predict(item, probe_threshold=min(1.0, confidence + 1e-6)).needs_probe)
        for bad in (-0.1, 1.5, float("nan")):
            with self.assertRaises(ValueError):
                self.classifier.predict(item, probe_threshold=bad)
        with self.assertRaises(TypeError):
            self.classifier.predict(item, probe_threshold="0.5")

    def test_forbidden_metadata_never_enters_classifier(self):
        offline = FORBIDDEN_RUNTIME_FIELDS | {"sample_id", "target_label", "metadata", "needs_manual_review"}
        self.assertFalse(offline & set(self.classifier.feature_names))
        record = {"input": {}, "target_label": "correct", "metadata": {"needs_manual_review": False}}
        with self.assertRaises(TypeError):
            DiagnosisClassifier.fit([record] * 4, list(LABEL_ORDER), c=1.0)
        item = self.other_inputs[0]
        renamed = dataclasses.replace(item, sample_id="different-id")
        np.testing.assert_array_equal(
            self.classifier.predict_proba_batch([item]), self.classifier.predict_proba_batch([renamed])
        )

    def test_invalid_feature_shape_raises(self):
        n = len(self.classifier.feature_names)
        for shape in ((1, n - 1), (1, n + 1), (n,)):
            with self.assertRaises(ValueError):
                self.classifier.predict_proba_matrix(np.zeros(shape))

    def test_training_requires_all_labels(self):
        with self.assertRaises(ValueError):
            DiagnosisClassifier.fit(self.train_inputs[:4], [DiagnosisLabel.CORRECT] * 4, c=1.0)
        with self.assertRaises(ValueError):
            DiagnosisClassifier.fit(self.train_inputs[:3], list(LABEL_ORDER)[:3], c=1.0)

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "diagnosis" / "model.json"
            self.classifier.save(path)
            loaded = DiagnosisClassifier.load(path)
        self.assertEqual(loaded.feature_names, self.classifier.feature_names)
        self.assertEqual(loaded.label_order, LABEL_ORDER)
        self.assertEqual(loaded.c, 1.0)
        self.assertTrue(loaded.include_refutes_prediction)
        self.assertEqual(loaded.random_state, 42)
        np.testing.assert_allclose(
            loaded.predict_proba_batch(self.other_inputs),
            self.classifier.predict_proba_batch(self.other_inputs),
            rtol=0, atol=1e-12,
        )
        for item in self.other_inputs:
            original, reloaded = self.classifier.predict(item), loaded.predict(item)
            self.assertIs(reloaded.diagnosis_label, original.diagnosis_label)
            self.assertAlmostEqual(reloaded.confidence, original.confidence, places=12)


if __name__ == "__main__":
    unittest.main()
