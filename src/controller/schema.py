"""Controller v1 runtime schema (routing only, no compensation).

A ControllerDecision is built only from runtime-visible Diagnosis, optional
Probe, and frozen SelectionDecision outputs. Gold answers/labels, target
labels, metadata, correctness, EM/F1, C0/C3, oracle and other evaluation
fields must never be added here. sample_id is optional and for logging only;
it never affects the decision. Runtime feature vectors are not part of this
public contract.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType

from src.controller.normalization import normalize_failure_type
from src.diagnosis.labels import DiagnosisLabel, parse_label

_FAILURE_SCORE_KEYS = (
    DiagnosisLabel.EVIDENCE_INSUFFICIENCY.value,
    DiagnosisLabel.EVIDENCE_ACCESSIBILITY.value,
    DiagnosisLabel.EVIDENCE_UTILISATION.value,
)


class ControllerAction(StrEnum):
    """Canonical Controller actions (repo names).

    External conceptual aliases (P62):
      accessibility_enhancement -> evidence_refinement
      utilisation_enhancement   -> generation_repair
    """

    NO_ACTION = "no_action"
    RETRIEVAL_ENHANCEMENT = "retrieval_enhancement"
    EVIDENCE_REFINEMENT = "evidence_refinement"
    GENERATION_REPAIR = "generation_repair"


# Reported mapping for docs / handoff (do not invent parallel enums).
ACTION_CONCEPTUAL_ALIASES: dict[str, str] = {
    "no_action": ControllerAction.NO_ACTION.value,
    "retrieval_enhancement": ControllerAction.RETRIEVAL_ENHANCEMENT.value,
    "accessibility_enhancement": ControllerAction.EVIDENCE_REFINEMENT.value,
    "utilisation_enhancement": ControllerAction.GENERATION_REPAIR.value,
}


class SelectionLane(StrEnum):
    """Frozen online adapter lane (P60). Selection is not Enhancement routing."""

    INSUFF_PRESERVE = "insuff_preserve"
    RESIDUAL_COMBO = "residual_combo"
    NO_ACTION = "no_action"


@dataclass(frozen=True)
class SelectionDecision:
    """Single-query selection output from FrozenOnlineSelectionAdapter.

    needs_enhancement gates whether Controller may later route to an Enhancement.
    This round does not invoke Enhancement modules. used_probe is always False
    on the frozen adapter path. selection_lane explains why a query was selected;
    it must not override failure_type for routing.

    FR6 representation (additive):
      primary_failure_type — single primary route label (defaults to failure_type)
      active_failures — zero or more concurrent failure subtypes
      failure_scores — runtime-safe joint subtype probabilities
      failure_type — backward-compatible alias of primary_failure_type
    """

    needs_enhancement: bool
    failure_type: DiagnosisLabel
    selection_score: float
    selection_lane: SelectionLane
    confidence: float
    p_fail: float
    p_insuff: float
    used_probe: bool = False
    sample_id: str | None = field(default=None, compare=False)
    d_combo: float | None = field(default=None, compare=False)
    rec: float | None = field(default=None, compare=False)
    p_corr_3c: float | None = field(default=None, compare=False)
    primary_failure_type: DiagnosisLabel | None = field(default=None)
    active_failures: tuple[DiagnosisLabel, ...] = field(default_factory=tuple)
    failure_scores: Mapping[str, float] = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        normalized = normalize_failure_type(self.failure_type)
        if normalized is None:
            raise ValueError(f"Invalid failure_type for SelectionDecision: {self.failure_type!r}")
        object.__setattr__(self, "failure_type", normalized)
        primary = self.primary_failure_type
        if primary is None:
            primary = normalized
        else:
            primary_n = normalize_failure_type(primary)
            if primary_n is None:
                raise ValueError(f"Invalid primary_failure_type: {primary!r}")
            primary = primary_n
        object.__setattr__(self, "primary_failure_type", primary)
        # Keep failure_type as alias of primary for backward compatibility.
        object.__setattr__(self, "failure_type", primary)
        object.__setattr__(self, "selection_lane", SelectionLane(self.selection_lane))
        if not isinstance(self.needs_enhancement, bool):
            raise TypeError("needs_enhancement must be bool")
        if not isinstance(self.used_probe, bool):
            raise TypeError("used_probe must be bool")
        if self.used_probe:
            raise ValueError("frozen online adapter requires used_probe=False")
        for name in ("confidence", "p_fail", "p_insuff"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a number, got {type(value).__name__}")
            value = float(value)
            if math.isnan(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0.0, 1.0], got {value}")
            object.__setattr__(self, name, value)
        score = self.selection_score
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise TypeError(f"selection_score must be a number, got {type(score).__name__}")
        if math.isnan(score) or math.isinf(score):
            raise ValueError(f"selection_score must be finite, got {score}")
        object.__setattr__(self, "selection_score", float(score))
        if self.needs_enhancement and self.selection_lane is SelectionLane.NO_ACTION:
            raise ValueError("needs_enhancement requires a non-no_action selection_lane")
        if not self.needs_enhancement and self.selection_lane is not SelectionLane.NO_ACTION:
            raise ValueError("selection_lane must be no_action when needs_enhancement is False")

        active = tuple(parse_label(x) for x in self.active_failures)
        for lab in active:
            if lab is DiagnosisLabel.CORRECT:
                raise ValueError("active_failures must not contain correct")
        if primary is DiagnosisLabel.CORRECT and active:
            raise ValueError("correct/no-failure requires empty active_failures")
        object.__setattr__(self, "active_failures", active)

        raw_scores = dict(self.failure_scores) if self.failure_scores else {}
        cleaned: dict[str, float] = {}
        for key in _FAILURE_SCORE_KEYS:
            val = float(raw_scores.get(key, 0.0))
            if math.isnan(val) or not 0.0 <= val <= 1.0:
                raise ValueError(f"failure_scores[{key!r}] must be in [0.0, 1.0], got {val}")
            cleaned[key] = val
        object.__setattr__(self, "failure_scores", MappingProxyType(cleaned))

    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "needs_enhancement": self.needs_enhancement,
            "failure_type": self.failure_type.value,
            "primary_failure_type": self.primary_failure_type.value
            if self.primary_failure_type is not None
            else self.failure_type.value,
            "active_failures": [lab.value for lab in self.active_failures],
            "failure_scores": dict(self.failure_scores),
            "selection_score": self.selection_score,
            "selection_lane": self.selection_lane.value,
            "confidence": self.confidence,
            "p_fail": self.p_fail,
            "p_insuff": self.p_insuff,
            "used_probe": self.used_probe,
            "d_combo": self.d_combo,
            "rec": self.rec,
            "p_corr_3c": self.p_corr_3c,
        }


@dataclass(frozen=True)
class ControllerDecision:
    """Routing decision (selection + optional Probe resolution).

    Public contract fields:
      action, needs_enhancement, failure_type (=diagnosis_label), selection_score,
      selection_lane, confidence (=diagnosis_confidence), used_probe,
      preserves_initial_answer, reason,
      primary_failure_type, active_failures, failure_scores (FR6).

    Routing still uses a single primary label → single action. active_failures is
    representation only; EnhancementExecutor is unchanged.

    needs_compensation remains for backward compatibility (== needs_enhancement).
    Runtime feature vectors are intentionally excluded.
    """

    action: ControllerAction
    diagnosis_label: DiagnosisLabel
    diagnosis_confidence: float
    used_probe: bool
    probe_overrode_diagnosis: bool
    sample_id: str | None = field(default=None, compare=False)
    selection_score: float | None = field(default=None, compare=False)
    selection_lane: SelectionLane | None = field(default=None, compare=False)
    reason: str | None = field(default=None, compare=False)
    primary_failure_type: DiagnosisLabel | None = field(default=None)
    active_failures: tuple[DiagnosisLabel, ...] = field(default_factory=tuple)
    failure_scores: Mapping[str, float] = field(default_factory=dict, hash=False)
    needs_compensation: bool = field(init=False)
    needs_enhancement: bool = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "action", ControllerAction(self.action))
        object.__setattr__(self, "diagnosis_label", parse_label(self.diagnosis_label))
        primary = self.primary_failure_type
        if primary is None:
            primary = self.diagnosis_label
        else:
            primary = parse_label(primary)
        object.__setattr__(self, "primary_failure_type", primary)
        confidence = self.diagnosis_confidence
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise TypeError(f"diagnosis_confidence must be a number, got {type(confidence).__name__}")
        if math.isnan(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError(f"diagnosis_confidence must be in [0.0, 1.0], got {confidence}")
        object.__setattr__(self, "diagnosis_confidence", float(confidence))
        for name in ("used_probe", "probe_overrode_diagnosis"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool")
        if self.probe_overrode_diagnosis and not self.used_probe:
            raise ValueError("probe_overrode_diagnosis requires used_probe")
        if self.selection_lane is not None:
            object.__setattr__(self, "selection_lane", SelectionLane(self.selection_lane))
        if self.selection_score is not None:
            score = self.selection_score
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise TypeError(f"selection_score must be a number, got {type(score).__name__}")
            if math.isnan(score) or math.isinf(score):
                raise ValueError(f"selection_score must be finite, got {score}")
            object.__setattr__(self, "selection_score", float(score))
        if self.reason is not None and not isinstance(self.reason, str):
            raise TypeError("reason must be str or None")
        active = tuple(parse_label(x) for x in self.active_failures)
        for lab in active:
            if lab is DiagnosisLabel.CORRECT:
                raise ValueError("active_failures must not contain correct")
        if primary is DiagnosisLabel.CORRECT and active:
            raise ValueError("correct/no-failure requires empty active_failures")
        object.__setattr__(self, "active_failures", active)
        raw_scores = dict(self.failure_scores) if self.failure_scores else {}
        cleaned: dict[str, float] = {}
        for key in _FAILURE_SCORE_KEYS:
            val = float(raw_scores.get(key, 0.0))
            if math.isnan(val) or not 0.0 <= val <= 1.0:
                raise ValueError(f"failure_scores[{key!r}] must be in [0.0, 1.0], got {val}")
            cleaned[key] = val
        object.__setattr__(self, "failure_scores", MappingProxyType(cleaned))
        needs = self.action is not ControllerAction.NO_ACTION
        object.__setattr__(self, "needs_compensation", needs)
        object.__setattr__(self, "needs_enhancement", needs)

    @property
    def failure_type(self) -> DiagnosisLabel:
        """Backward-compatible alias of primary_failure_type / diagnosis_label."""
        return self.primary_failure_type if self.primary_failure_type is not None else self.diagnosis_label

    @property
    def confidence(self) -> float:
        return self.diagnosis_confidence

    @property
    def preserves_initial_answer(self) -> bool:
        """True iff no Enhancement may run; host must keep Reference RAG answer/evidence."""
        return self.action is ControllerAction.NO_ACTION

    def to_dict(self) -> dict:
        primary = self.primary_failure_type if self.primary_failure_type is not None else self.diagnosis_label
        return {
            "sample_id": self.sample_id,
            "action": self.action.value,
            "diagnosis_label": self.diagnosis_label.value,
            "failure_type": self.failure_type.value,
            "primary_failure_type": primary.value,
            "active_failures": [lab.value for lab in self.active_failures],
            "failure_scores": dict(self.failure_scores),
            "diagnosis_confidence": self.diagnosis_confidence,
            "confidence": self.confidence,
            "used_probe": self.used_probe,
            "probe_overrode_diagnosis": self.probe_overrode_diagnosis,
            "needs_compensation": self.needs_compensation,
            "needs_enhancement": self.needs_enhancement,
            "selection_score": self.selection_score,
            "selection_lane": None if self.selection_lane is None else self.selection_lane.value,
            "reason": self.reason,
            "preserves_initial_answer": self.preserves_initial_answer,
        }
