"""Controller v1 runtime schema (routing only, no compensation).

A ControllerDecision is built only from runtime-visible Diagnosis and Probe
outputs. Gold answers/labels, target labels, metadata, correctness, EM/F1,
C0/C3, oracle and other evaluation fields must never be added here.
sample_id is optional and for logging only; it never affects the decision.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum

from src.diagnosis.labels import DiagnosisLabel, parse_label


class ControllerAction(StrEnum):
    NO_ACTION = "no_action"
    RETRIEVAL_ENHANCEMENT = "retrieval_enhancement"
    EVIDENCE_REFINEMENT = "evidence_refinement"
    GENERATION_REPAIR = "generation_repair"


@dataclass(frozen=True)
class ControllerDecision:
    """Routing decision.

    diagnosis_label is the resolved label after Safe Binary resolution;
    diagnosis_confidence is the confidence of the original Diagnosis decision.
    needs_compensation is derived as action != NO_ACTION.
    """

    action: ControllerAction
    diagnosis_label: DiagnosisLabel
    diagnosis_confidence: float
    used_probe: bool
    probe_overrode_diagnosis: bool
    sample_id: str | None = field(default=None, compare=False)
    needs_compensation: bool = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "action", ControllerAction(self.action))
        object.__setattr__(self, "diagnosis_label", parse_label(self.diagnosis_label))
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
        object.__setattr__(self, "needs_compensation", self.action is not ControllerAction.NO_ACTION)

    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "action": self.action.value,
            "diagnosis_label": self.diagnosis_label.value,
            "diagnosis_confidence": self.diagnosis_confidence,
            "used_probe": self.used_probe,
            "probe_overrode_diagnosis": self.probe_overrode_diagnosis,
            "needs_compensation": self.needs_compensation,
        }
