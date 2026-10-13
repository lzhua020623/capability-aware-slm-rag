"""Hierarchical Diagnosis: Stage 1 correct-vs-failure, Stage 2 failure type.

Stage 1 is a binary Logistic Regression producing P(failure); a sample is a
failure when P(failure) >= stage1_threshold. Only Stage 1 failures are passed
to Stage 2, a three-class Logistic Regression over the failure labels. Both
stages read only extract_features(DiagnosisInput) in the canonical feature
order, and each scaler is fitted only on the samples given to fit().

Final class_scores are the joint probabilities
P(correct) = 1 - P(failure) and P(type) = P(failure) * P(type | failure);
they are reported only when Stage 2 runs. confidence is the probability of
the final label: 1 - P(failure) for correct, the joint probability otherwise.

The single-stage baseline in src/diagnosis/classifier.py is unchanged.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.diagnosis.classifier import (
    LABEL_ORDER,
    MAX_ITER,
    RANDOM_STATE,
    _validate_probe_threshold,
    feature_matrix,
    feature_names_for,
)
from src.diagnosis.labels import DiagnosisLabel, parse_label
from src.diagnosis.schema import DiagnosisInput, DiagnosisOutput

ARTIFACT_VERSION = "diagnosis_hierarchical_v1"
STAGE1_LABELS: tuple[str, ...] = ("correct", "failure")
FAILURE_LABELS: tuple[DiagnosisLabel, ...] = tuple(label for label in LABEL_ORDER if label is not DiagnosisLabel.CORRECT)
CLASS_WEIGHTS: tuple[str | None, ...] = (None, "balanced")


def _validate_stage1_threshold(threshold: float) -> float:
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise TypeError(f"stage1_threshold must be a number, got {type(threshold).__name__}")
    if math.isnan(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError(f"stage1_threshold must be in [0.0, 1.0], got {threshold}")
    return float(threshold)


@dataclass
class LogisticStage:
    """StandardScaler + L2 Logistic Regression over integer targets indexing `labels`."""

    scaler: StandardScaler
    model: LogisticRegression
    labels: tuple[str, ...]
    c: float
    class_weight: str | None
    random_state: int

    @classmethod
    def fit(
        cls,
        matrix: np.ndarray,
        targets: Sequence[int],
        labels: Sequence[str],
        c: float,
        class_weight: str | None,
        random_state: int = RANDOM_STATE,
        max_iter: int = MAX_ITER,
    ) -> LogisticStage:
        targets = np.asarray(targets, dtype=int)
        if matrix.shape[0] != targets.shape[0]:
            raise ValueError(f"Got {matrix.shape[0]} rows but {targets.shape[0]} targets")
        if class_weight not in CLASS_WEIGHTS:
            raise ValueError(f"class_weight must be one of {CLASS_WEIGHTS}, got {class_weight!r}")
        missing = set(range(len(labels))) - set(targets.tolist())
        if missing:
            raise ValueError(f"Training targets are missing classes: {sorted(labels[i] for i in missing)}")
        scaler = StandardScaler().fit(matrix)
        model = LogisticRegression(
            C=c,
            l1_ratio=0.0,
            class_weight=class_weight,
            solver="lbfgs",
            max_iter=max_iter,
            random_state=random_state,
        ).fit(scaler.transform(matrix), targets)
        return cls(scaler, model, tuple(labels), float(c), class_weight, random_state)

    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=float)
        expected = self.scaler.n_features_in_
        if matrix.ndim != 2 or matrix.shape[1] != expected:
            raise ValueError(f"Expected feature matrix of shape (n, {expected}), got {matrix.shape}")
        return self.model.predict_proba(self.scaler.transform(matrix))

    def to_dict(self) -> dict:
        return {
            "labels": list(self.labels),
            "c": self.c,
            "class_weight": self.class_weight,
            "random_state": self.random_state,
            "scaler": {
                "mean": self.scaler.mean_.tolist(),
                "scale": self.scaler.scale_.tolist(),
                "var": self.scaler.var_.tolist(),
                "n_samples_seen": int(self.scaler.n_samples_seen_),
            },
            "model": {
                "coef": self.model.coef_.tolist(),
                "intercept": self.model.intercept_.tolist(),
                "classes": self.model.classes_.tolist(),
                "max_iter": self.model.max_iter,
                "n_iter": self.model.n_iter_.tolist(),
            },
        }

    @classmethod
    def from_dict(cls, data: dict, n_features: int) -> LogisticStage:
        scaler = StandardScaler()
        scaler.mean_ = np.asarray(data["scaler"]["mean"], dtype=float)
        scaler.scale_ = np.asarray(data["scaler"]["scale"], dtype=float)
        scaler.var_ = np.asarray(data["scaler"]["var"], dtype=float)
        scaler.n_samples_seen_ = data["scaler"]["n_samples_seen"]
        scaler.n_features_in_ = n_features

        model = LogisticRegression(
            C=data["c"],
            l1_ratio=0.0,
            class_weight=data["class_weight"],
            solver="lbfgs",
            max_iter=data["model"]["max_iter"],
            random_state=data["random_state"],
        )
        model.coef_ = np.asarray(data["model"]["coef"], dtype=float)
        model.intercept_ = np.asarray(data["model"]["intercept"], dtype=float)
        model.classes_ = np.asarray(data["model"]["classes"])
        model.n_iter_ = np.asarray(data["model"]["n_iter"])
        model.n_features_in_ = n_features
        if model.coef_.shape[1] != n_features or scaler.mean_.shape != (n_features,):
            raise ValueError("Stage parameters do not match the feature count")
        return cls(scaler, model, tuple(data["labels"]), float(data["c"]), data["class_weight"], int(data["random_state"]))


@dataclass(frozen=True)
class HierarchicalDiagnosis:
    """Full hierarchical decision, kept for later Probe design."""

    stage1_failure_probability: float
    stage1_label: str
    stage2_class_scores: dict[DiagnosisLabel, float] | None
    final_label: DiagnosisLabel
    confidence: float
    class_scores: dict[DiagnosisLabel, float] | None

    def to_output(self, probe_threshold: float | None = None) -> DiagnosisOutput:
        threshold = _validate_probe_threshold(probe_threshold)
        return DiagnosisOutput(
            diagnosis_label=self.final_label,
            confidence=self.confidence,
            needs_probe=threshold is not None and self.confidence < threshold,
            class_scores=self.class_scores,
        )


@dataclass
class HierarchicalDiagnosisClassifier:
    stage1: LogisticStage
    stage2: LogisticStage
    feature_names: tuple[str, ...]
    stage1_threshold: float
    include_refutes_prediction: bool
    random_state: int

    def __post_init__(self) -> None:
        self.stage1_threshold = _validate_stage1_threshold(self.stage1_threshold)
        if self.feature_names != feature_names_for(self.include_refutes_prediction):
            raise ValueError("feature_names do not match the canonical feature order")
        if self.stage1.labels != STAGE1_LABELS:
            raise ValueError(f"Stage 1 labels must be {STAGE1_LABELS}, got {self.stage1.labels}")
        if self.stage2.labels != tuple(label.value for label in FAILURE_LABELS):
            raise ValueError(f"Stage 2 labels must be the failure labels, got {self.stage2.labels}")

    @classmethod
    def fit(
        cls,
        inputs: Sequence[DiagnosisInput],
        labels: Sequence[DiagnosisLabel | str],
        stage1_c: float,
        stage1_class_weight: str | None,
        stage2_c: float,
        stage1_threshold: float = 0.5,
        include_refutes_prediction: bool = True,
        random_state: int = RANDOM_STATE,
    ) -> HierarchicalDiagnosisClassifier:
        """Stage 1 is fitted on all given samples, Stage 2 on the failure samples only."""
        if len(inputs) != len(labels):
            raise ValueError(f"Got {len(inputs)} inputs but {len(labels)} labels")
        parsed = [parse_label(label) for label in labels]
        feature_names = feature_names_for(include_refutes_prediction)
        matrix = feature_matrix(inputs, feature_names)
        is_failure = np.asarray([label is not DiagnosisLabel.CORRECT for label in parsed])
        stage1 = LogisticStage.fit(
            matrix, is_failure.astype(int), STAGE1_LABELS, stage1_c, stage1_class_weight, random_state
        )
        stage2 = LogisticStage.fit(
            matrix[is_failure],
            [FAILURE_LABELS.index(label) for label in parsed if label is not DiagnosisLabel.CORRECT],
            [label.value for label in FAILURE_LABELS],
            stage2_c,
            "balanced",
            random_state,
        )
        return cls(stage1, stage2, feature_names, stage1_threshold, include_refutes_prediction, random_state)

    def stage1_failure_probability(self, matrix: np.ndarray) -> np.ndarray:
        return self.stage1.predict_proba(matrix)[:, STAGE1_LABELS.index("failure")]

    def diagnose_matrix(self, matrix: np.ndarray) -> list[HierarchicalDiagnosis]:
        failure_probability = self.stage1_failure_probability(matrix)
        is_failure = failure_probability >= self.stage1_threshold
        stage2_proba = self.stage2.predict_proba(np.asarray(matrix)[is_failure]) if is_failure.any() else None

        results, stage2_row = [], 0
        for p_failure, failure in zip(failure_probability.tolist(), is_failure.tolist()):
            if not failure:
                results.append(HierarchicalDiagnosis(
                    stage1_failure_probability=p_failure,
                    stage1_label="correct",
                    stage2_class_scores=None,
                    final_label=DiagnosisLabel.CORRECT,
                    confidence=1.0 - p_failure,
                    class_scores=None,
                ))
                continue
            conditional = {label: float(p) for label, p in zip(FAILURE_LABELS, stage2_proba[stage2_row])}
            stage2_row += 1
            final = max(FAILURE_LABELS, key=conditional.get)
            joint = {DiagnosisLabel.CORRECT: 1.0 - p_failure}
            joint.update({label: p_failure * p for label, p in conditional.items()})
            results.append(HierarchicalDiagnosis(
                stage1_failure_probability=p_failure,
                stage1_label="failure",
                stage2_class_scores=conditional,
                final_label=final,
                confidence=joint[final],
                class_scores=joint,
            ))
        return results

    def diagnose_batch(self, items: Sequence[DiagnosisInput]) -> list[HierarchicalDiagnosis]:
        return self.diagnose_matrix(feature_matrix(items, self.feature_names))

    def diagnose(self, item: DiagnosisInput) -> HierarchicalDiagnosis:
        return self.diagnose_batch([item])[0]

    def predict(self, item: DiagnosisInput, probe_threshold: float | None = None) -> DiagnosisOutput:
        """Runtime API compatible with the single-stage classifier; needs_probe defaults to False."""
        return self.diagnose(item).to_output(probe_threshold)

    def to_dict(self) -> dict:
        return {
            "version": ARTIFACT_VERSION,
            "feature_names": list(self.feature_names),
            "label_order": [label.value for label in LABEL_ORDER],
            "stage1_threshold": self.stage1_threshold,
            "include_refutes_prediction": self.include_refutes_prediction,
            "random_state": self.random_state,
            "stage1": self.stage1.to_dict(),
            "stage2": self.stage2.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> HierarchicalDiagnosisClassifier:
        if data.get("version") != ARTIFACT_VERSION:
            raise ValueError(f"Unsupported artifact version {data.get('version')!r}")
        if tuple(parse_label(label) for label in data["label_order"]) != LABEL_ORDER:
            raise ValueError("Artifact label order does not match the runtime label order")
        feature_names = tuple(data["feature_names"])
        return cls(
            stage1=LogisticStage.from_dict(data["stage1"], len(feature_names)),
            stage2=LogisticStage.from_dict(data["stage2"], len(feature_names)),
            feature_names=feature_names,
            stage1_threshold=data["stage1_threshold"],
            include_refutes_prediction=bool(data["include_refutes_prediction"]),
            random_state=int(data["random_state"]),
        )

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> HierarchicalDiagnosisClassifier:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
