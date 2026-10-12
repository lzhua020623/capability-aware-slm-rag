"""Diagnosis v1 classifier: StandardScaler + multinomial Logistic Regression.

The classifier sees only the runtime feature vector produced by
extract_features(DiagnosisInput), in an explicit, fixed feature order.
fit() accepts DiagnosisInput objects and labels only, so the scaler and the
model are fitted on exactly the samples passed in (the train split); other
splits must go through predict/predict_proba, which only transform.

The artifact is plain JSON (parameters, feature names, label order and
training configuration); no pickle is used.
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

from src.diagnosis.features import FEATURE_NAMES, extract_features
from src.diagnosis.labels import RUNTIME_LABELS, DiagnosisLabel, parse_label
from src.diagnosis.schema import DiagnosisInput, DiagnosisOutput

ARTIFACT_VERSION = "diagnosis_v1"
LABEL_ORDER: tuple[DiagnosisLabel, ...] = RUNTIME_LABELS
REFUTES_FEATURE = "is_refutes_prediction"
RANDOM_STATE = 42
MAX_ITER = 5000


def feature_names_for(include_refutes_prediction: bool) -> tuple[str, ...]:
    """Canonical feature order: FEATURE_NAMES, optionally without is_refutes_prediction."""
    if include_refutes_prediction:
        return FEATURE_NAMES
    return tuple(name for name in FEATURE_NAMES if name != REFUTES_FEATURE)


def feature_matrix(items: Sequence[DiagnosisInput], feature_names: Sequence[str]) -> np.ndarray:
    rows = []
    for item in items:
        if not isinstance(item, DiagnosisInput):
            raise TypeError(f"Expected DiagnosisInput, got {type(item).__name__}")
        features = extract_features(item)
        rows.append([features[name] for name in feature_names])
    return np.asarray(rows, dtype=float).reshape(len(rows), len(feature_names))


def _validate_probe_threshold(probe_threshold: float | None) -> float | None:
    if probe_threshold is None:
        return None
    if isinstance(probe_threshold, bool) or not isinstance(probe_threshold, (int, float)):
        raise TypeError(f"probe_threshold must be a number or None, got {type(probe_threshold).__name__}")
    if math.isnan(probe_threshold) or not 0.0 <= probe_threshold <= 1.0:
        raise ValueError(f"probe_threshold must be in [0.0, 1.0], got {probe_threshold}")
    return float(probe_threshold)


@dataclass
class DiagnosisClassifier:
    scaler: StandardScaler
    model: LogisticRegression
    feature_names: tuple[str, ...]
    label_order: tuple[DiagnosisLabel, ...]
    c: float
    include_refutes_prediction: bool
    random_state: int

    @classmethod
    def fit(
        cls,
        inputs: Sequence[DiagnosisInput],
        labels: Sequence[DiagnosisLabel | str],
        c: float,
        include_refutes_prediction: bool = True,
        random_state: int = RANDOM_STATE,
        max_iter: int = MAX_ITER,
    ) -> DiagnosisClassifier:
        """Fit scaler and model on the given (train) inputs only."""
        if len(inputs) != len(labels):
            raise ValueError(f"Got {len(inputs)} inputs but {len(labels)} labels")
        parsed = [parse_label(label) for label in labels]
        missing = set(LABEL_ORDER) - set(parsed)
        if missing:
            raise ValueError(f"Training labels are missing classes: {sorted(missing)}")

        feature_names = feature_names_for(include_refutes_prediction)
        matrix = feature_matrix(inputs, feature_names)
        targets = np.asarray([LABEL_ORDER.index(label) for label in parsed])

        scaler = StandardScaler().fit(matrix)
        model = LogisticRegression(
            C=c,
            l1_ratio=0.0,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=max_iter,
            random_state=random_state,
        ).fit(scaler.transform(matrix), targets)
        return cls(
            scaler=scaler,
            model=model,
            feature_names=feature_names,
            label_order=LABEL_ORDER,
            c=float(c),
            include_refutes_prediction=include_refutes_prediction,
            random_state=random_state,
        )

    def predict_proba_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Class probabilities in label_order; the scaler only transforms here."""
        matrix = np.asarray(matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Expected feature matrix of shape (n, {len(self.feature_names)}), got {matrix.shape}"
            )
        return self.model.predict_proba(self.scaler.transform(matrix))

    def predict_proba_batch(self, items: Sequence[DiagnosisInput]) -> np.ndarray:
        return self.predict_proba_matrix(feature_matrix(items, self.feature_names))

    def output_from_proba(self, proba: np.ndarray, probe_threshold: float | None = None) -> DiagnosisOutput:
        threshold = _validate_probe_threshold(probe_threshold)
        best = int(np.argmax(proba))
        confidence = float(proba[best])
        return DiagnosisOutput(
            diagnosis_label=self.label_order[best],
            confidence=confidence,
            needs_probe=threshold is not None and confidence < threshold,
            class_scores={label: float(score) for label, score in zip(self.label_order, proba)},
        )

    def predict(self, item: DiagnosisInput, probe_threshold: float | None = None) -> DiagnosisOutput:
        """Runtime API. needs_probe is False unless a probe_threshold is given."""
        return self.output_from_proba(self.predict_proba_batch([item])[0], probe_threshold)

    def to_dict(self) -> dict:
        return {
            "version": ARTIFACT_VERSION,
            "feature_names": list(self.feature_names),
            "label_order": [label.value for label in self.label_order],
            "c": self.c,
            "include_refutes_prediction": self.include_refutes_prediction,
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
    def from_dict(cls, data: dict) -> DiagnosisClassifier:
        if data.get("version") != ARTIFACT_VERSION:
            raise ValueError(f"Unsupported artifact version {data.get('version')!r}")
        feature_names = tuple(data["feature_names"])
        if feature_names != feature_names_for(data["include_refutes_prediction"]):
            raise ValueError("Artifact feature order does not match the canonical feature order")
        label_order = tuple(parse_label(label) for label in data["label_order"])
        if label_order != LABEL_ORDER:
            raise ValueError("Artifact label order does not match the runtime label order")

        scaler = StandardScaler()
        scaler.mean_ = np.asarray(data["scaler"]["mean"], dtype=float)
        scaler.scale_ = np.asarray(data["scaler"]["scale"], dtype=float)
        scaler.var_ = np.asarray(data["scaler"]["var"], dtype=float)
        scaler.n_samples_seen_ = data["scaler"]["n_samples_seen"]
        scaler.n_features_in_ = len(feature_names)

        model = LogisticRegression(
            C=data["c"],
            l1_ratio=0.0,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=data["model"]["max_iter"],
            random_state=data["random_state"],
        )
        model.coef_ = np.asarray(data["model"]["coef"], dtype=float)
        model.intercept_ = np.asarray(data["model"]["intercept"], dtype=float)
        model.classes_ = np.asarray(data["model"]["classes"])
        model.n_iter_ = np.asarray(data["model"]["n_iter"])
        model.n_features_in_ = len(feature_names)
        if model.coef_.shape != (len(label_order), len(feature_names)):
            raise ValueError(f"Artifact coef shape {model.coef_.shape} does not match features/labels")

        return cls(
            scaler=scaler,
            model=model,
            feature_names=feature_names,
            label_order=label_order,
            c=float(data["c"]),
            include_refutes_prediction=bool(data["include_refutes_prediction"]),
            random_state=int(data["random_state"]),
        )

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> DiagnosisClassifier:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
