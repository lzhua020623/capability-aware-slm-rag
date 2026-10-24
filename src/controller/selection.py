"""Frozen online two-lane selection adapter (P60 formalised).

Single-query only. No batch ranking APIs, no gold, no Probe, no Enhancement.
Thresholds, z-stats and HGB artifacts are frozen from EXP-P60; this module does
not recalibrate.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.controller.failure_representation import representation_from_diagnosis
from src.controller.schema import SelectionDecision, SelectionLane
from src.diagnosis.hierarchical_classifier import HierarchicalDiagnosis
from src.diagnosis.labels import DiagnosisLabel

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "controller" / "p60_online_adapter.json"


def _unit(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    value = float(value)
    if math.isnan(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0.0, 1.0], got {value}")
    return value


def _finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        raise ValueError(f"{name} must be finite, got {value}")
    return value


def apply_frozen_z(x: float, mean: float, std: float) -> float:
    if std < 1e-12:
        return 0.0
    return (x - mean) / std


def p_insuff_from_diagnosis(diagnosis: HierarchicalDiagnosis) -> float:
    """Joint P(evidence_insufficiency) from formal hierarchical outputs."""
    if diagnosis.class_scores is not None:
        return _unit(
            "p_insuff",
            diagnosis.class_scores.get(DiagnosisLabel.EVIDENCE_INSUFFICIENCY, 0.0),
        )
    if diagnosis.stage2_class_scores is not None:
        conditional = diagnosis.stage2_class_scores.get(
            DiagnosisLabel.EVIDENCE_INSUFFICIENCY, 0.0
        )
        return _unit(
            "p_insuff",
            diagnosis.stage1_failure_probability * float(conditional),
        )
    return 0.0


def d_combo_score(
    p_fail: float,
    rec: float,
    p_corr: float,
    *,
    alpha: float,
    lam: float,
    z_stats: dict[str, dict[str, float]],
) -> float:
    zf = apply_frozen_z(p_fail, z_stats["p_fail"]["mean"], z_stats["p_fail"]["std"])
    zr = apply_frozen_z(rec, z_stats["rec"]["mean"], z_stats["rec"]["std"])
    zc = apply_frozen_z(p_corr, z_stats["p_corr"]["mean"], z_stats["p_corr"]["std"])
    return alpha * zf + (1.0 - alpha) * zr - lam * zc


@dataclass(frozen=True)
class FrozenAdapterModels:
    fail_gate: Any
    rec_given_fail: Any | None
    multiclass_3c: Any


class FrozenOnlineSelectionAdapter:
    """P60 C_two_lane_balanced online adapter: Lane1 p_insuff, Lane2 frozen D_combo."""

    def __init__(
        self,
        config: dict[str, Any],
        models: FrozenAdapterModels,
        *,
        config_path: Path | None = None,
    ) -> None:
        self.config = dict(config)
        self.models = models
        self.config_path = config_path
        self.version = str(self.config.get("version", ""))
        self.probe_enabled = bool(self.config.get("probe_enabled", False))
        if self.probe_enabled:
            raise ValueError("Frozen online adapter requires probe_enabled=false")
        thresholds = self.config["thresholds"]
        self.t_insuff = _finite("t_insuff", thresholds["t_insuff"])
        self.t_combo = _finite("t_combo", thresholds["t_combo"])
        d_combo = self.config["d_combo"]
        self.alpha = _finite("alpha", d_combo["alpha"])
        self.lam = _finite("lam", d_combo["lam"])
        self.z_stats = {
            key: {
                "mean": _finite(f"z_stats.{key}.mean", stats["mean"]),
                "std": _finite(f"z_stats.{key}.std", stats["std"]),
            }
            for key, stats in self.config["z_stats"].items()
        }
        self.feature_names: tuple[str, ...] = tuple(self.config["feature_names"])
        self.n_features = len(self.feature_names)
        self.p_corr_class_index = int(self.config.get("p_corr_class_index", 2))
        expected = getattr(models.fail_gate, "n_features_in_", None)
        if expected is not None and int(expected) != self.n_features:
            raise ValueError(
                f"Config has {self.n_features} feature_names but fail_gate expects {expected}"
            )

    @classmethod
    def from_config_path(
        cls,
        config_path: Path | str | None = None,
        *,
        repo_root: Path | None = None,
    ) -> FrozenOnlineSelectionAdapter:
        root = REPO_ROOT if repo_root is None else Path(repo_root)
        path = DEFAULT_CONFIG_PATH if config_path is None else Path(config_path)
        if not path.is_absolute():
            path = root / path
        config = json.loads(path.read_text(encoding="utf-8"))
        models = _load_models(config["models"], root)
        return cls(config, models, config_path=path)

    @classmethod
    def from_frozen_defaults(cls, repo_root: Path | None = None) -> FrozenOnlineSelectionAdapter:
        return cls.from_config_path(DEFAULT_CONFIG_PATH, repo_root=repo_root)

    def _validate_features(self, adapter_features: np.ndarray | list[float]) -> np.ndarray:
        row = np.asarray(adapter_features, dtype=float).reshape(1, -1)
        if row.shape[1] != self.n_features:
            raise ValueError(
                f"adapter_features must have length {self.n_features}, got {row.shape[1]}"
            )
        if not np.isfinite(row).all():
            raise ValueError("adapter_features must be finite")
        return row

    def _predict_rec_pcorr(self, features: np.ndarray) -> tuple[float, float]:
        row = self._validate_features(features)
        p_fail_stage = _positive_class_proba(self.models.fail_gate, row, positive=1)
        if self.models.rec_given_fail is None:
            p_rec_g = 0.0
        else:
            p_rec_g = _positive_class_proba(self.models.rec_given_fail, row, positive=1)
        rec = float(p_fail_stage * p_rec_g)
        proba = self.models.multiclass_3c.predict_proba(row)[0]
        classes = [int(c) for c in self.models.multiclass_3c.classes_]
        if self.p_corr_class_index not in classes:
            p_corr = 0.0
        else:
            p_corr = float(proba[classes.index(self.p_corr_class_index)])
        return rec, p_corr

    def select(
        self,
        diagnosis: HierarchicalDiagnosis,
        adapter_features: np.ndarray | list[float],
        *,
        sample_id: str | None = None,
        p_fail: float | None = None,
        p_insuff: float | None = None,
        rec: float | None = None,
        p_corr_3c: float | None = None,
    ) -> SelectionDecision:
        """Single-query selection. Optional score overrides are for tests/regression only."""
        if not isinstance(diagnosis, HierarchicalDiagnosis):
            raise TypeError(
                f"diagnosis must be HierarchicalDiagnosis, got {type(diagnosis).__name__}"
            )
        self._validate_features(adapter_features)

        fr6 = representation_from_diagnosis(diagnosis)

        # Correct Diagnosis never enters enhancement lanes (P57/P60 pred-fail pool semantics).
        if diagnosis.final_label is DiagnosisLabel.CORRECT or diagnosis.stage1_label == "correct":
            p_fail_v = _unit(
                "p_fail",
                diagnosis.stage1_failure_probability if p_fail is None else p_fail,
            )
            p_insuff_v = 0.0 if p_insuff is None else _unit("p_insuff", p_insuff)
            return SelectionDecision(
                needs_enhancement=False,
                failure_type=DiagnosisLabel.CORRECT,
                selection_score=0.0,
                selection_lane=SelectionLane.NO_ACTION,
                confidence=_unit("confidence", diagnosis.confidence),
                p_fail=p_fail_v,
                p_insuff=p_insuff_v,
                used_probe=False,
                sample_id=sample_id,
                d_combo=None,
                rec=None,
                p_corr_3c=None,
                primary_failure_type=fr6.primary_failure_type,
                active_failures=fr6.active_failures,
                failure_scores=fr6.failure_scores,
            )

        p_fail_v = _unit(
            "p_fail",
            diagnosis.stage1_failure_probability if p_fail is None else p_fail,
        )
        p_insuff_v = (
            p_insuff_from_diagnosis(diagnosis)
            if p_insuff is None
            else _unit("p_insuff", p_insuff)
        )

        if rec is None or p_corr_3c is None:
            pred_rec, pred_corr = self._predict_rec_pcorr(adapter_features)
            rec_v = pred_rec if rec is None else _finite("rec", rec)
            corr_v = pred_corr if p_corr_3c is None else _finite("p_corr_3c", p_corr_3c)
        else:
            rec_v = _finite("rec", rec)
            corr_v = _finite("p_corr_3c", p_corr_3c)

        combo = d_combo_score(
            p_fail_v,
            rec_v,
            corr_v,
            alpha=self.alpha,
            lam=self.lam,
            z_stats=self.z_stats,
        )

        if p_insuff_v >= self.t_insuff:
            lane = SelectionLane.INSUFF_PRESERVE
            score = p_insuff_v
            selected = True
        elif combo >= self.t_combo:
            lane = SelectionLane.RESIDUAL_COMBO
            score = combo
            selected = True
        else:
            lane = SelectionLane.NO_ACTION
            score = combo
            selected = False

        return SelectionDecision(
            needs_enhancement=selected,
            failure_type=diagnosis.final_label,
            selection_score=float(score),
            selection_lane=lane,
            confidence=_unit("confidence", diagnosis.confidence),
            p_fail=p_fail_v,
            p_insuff=p_insuff_v,
            used_probe=False,
            sample_id=sample_id,
            d_combo=float(combo),
            rec=float(rec_v),
            p_corr_3c=float(corr_v),
            primary_failure_type=fr6.primary_failure_type,
            active_failures=fr6.active_failures,
            failure_scores=fr6.failure_scores,
        )


def _load_models(model_paths: dict[str, str], repo_root: Path) -> FrozenAdapterModels:
    try:
        import joblib
    except ImportError as exc:  # pragma: no cover
        raise ImportError("joblib is required to load frozen selection models") from exc

    def resolve(key: str):
        rel = model_paths[key]
        path = Path(rel)
        if not path.is_absolute():
            path = repo_root / path
        if not path.is_file():
            raise FileNotFoundError(f"Missing frozen adapter model {key}: {path}")
        return joblib.load(path)

    return FrozenAdapterModels(
        fail_gate=resolve("hgb_fail_gate"),
        rec_given_fail=resolve("hgb_rec_given_fail"),
        multiclass_3c=resolve("hgb_3c"),
    )


def _positive_class_proba(model: Any, row: np.ndarray, positive: int = 1) -> float:
    proba = model.predict_proba(row)[0]
    classes = [int(c) for c in model.classes_]
    if positive not in classes:
        return 0.0
    return float(proba[classes.index(positive)])
