"""Controller v1: Diagnosis -> optional Probe -> Safe Binary resolution -> routing.

Routing only: the returned action names a later compensation module, but no
retriever, reranker, generator or model is called here. All functions are
deterministic and read only runtime-visible Diagnosis / Probe outputs.

Safe Binary resolution (development policy frozen on the v2-expanded
validation split): the Probe may only turn a Diagnosis failure back into
correct, and only when the Probe says correct, Stage 1 P(failure) <=
SAFE_OVERRIDE_MAX_STAGE1 and max Stage 2 score <= SAFE_OVERRIDE_MAX_STAGE2.
The Probe never changes one failure type into another.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from src.controller.schema import ControllerAction, ControllerDecision
from src.diagnosis.hierarchical_classifier import HierarchicalDiagnosis, HierarchicalDiagnosisClassifier
from src.diagnosis.labels import DiagnosisLabel, parse_label
from src.diagnosis.schema import DiagnosisInput
from src.probe.pipeline import ProbedDiagnosis, apply_probe
from src.probe.probe import EvidenceProbe
from src.probe.schema import ProbeLabel, ProbeResult
from src.probe.trigger import V2_EXPANDED_TRIGGER, TriggerPolicy

SAFE_OVERRIDE_MAX_STAGE1 = 0.70
SAFE_OVERRIDE_MAX_STAGE2 = 0.50

ACTION_BY_LABEL: Mapping[DiagnosisLabel, ControllerAction] = {
    DiagnosisLabel.CORRECT: ControllerAction.NO_ACTION,
    DiagnosisLabel.EVIDENCE_INSUFFICIENCY: ControllerAction.RETRIEVAL_ENHANCEMENT,
    DiagnosisLabel.EVIDENCE_ACCESSIBILITY: ControllerAction.EVIDENCE_REFINEMENT,
    DiagnosisLabel.EVIDENCE_UTILISATION: ControllerAction.GENERATION_REPAIR,
}


def _probability(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    value = float(value)
    if math.isnan(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0.0, 1.0], got {value}")
    return value


def resolve_final_diagnosis(
    diagnosis_label: DiagnosisLabel | str,
    stage1_failure_probability: float,
    stage2_class_scores: Mapping[object, float] | None,
    probe_triggered: bool,
    probe_result: ProbeResult | None,
) -> DiagnosisLabel:
    """Safe Binary resolution of the Diagnosis label with an optional Probe result."""
    label = parse_label(diagnosis_label)
    p_failure = _probability("stage1_failure_probability", stage1_failure_probability)
    if not isinstance(probe_triggered, bool):
        raise TypeError("probe_triggered must be bool")
    if probe_triggered != (probe_result is not None):
        raise ValueError("probe_result must be given exactly when the Probe was triggered")
    if probe_result is None or label is DiagnosisLabel.CORRECT:
        return label
    if not isinstance(probe_result, ProbeResult):
        raise TypeError(f"probe_result must be a ProbeResult, got {type(probe_result).__name__}")
    if probe_result.probe_label is not ProbeLabel.CORRECT:
        return label
    if p_failure > SAFE_OVERRIDE_MAX_STAGE1 or not stage2_class_scores:
        return label
    stage2_max = max(_probability("stage2_class_scores", v) for v in stage2_class_scores.values())
    return DiagnosisLabel.CORRECT if stage2_max <= SAFE_OVERRIDE_MAX_STAGE2 else label


def route_controller(final_label: DiagnosisLabel | str) -> ControllerAction:
    """Strict label -> action mapping; unknown labels raise ValueError."""
    return ACTION_BY_LABEL[parse_label(final_label)]


def decide_action(
    diagnosis: HierarchicalDiagnosis,
    probe_triggered: bool,
    probe_result: ProbeResult | None,
    sample_id: str | None = None,
) -> ControllerDecision:
    """Safe Binary resolution followed by routing. sample_id is logged only."""
    resolved = resolve_final_diagnosis(
        diagnosis.final_label,
        diagnosis.stage1_failure_probability,
        diagnosis.stage2_class_scores,
        probe_triggered,
        probe_result,
    )
    return ControllerDecision(
        action=route_controller(resolved),
        diagnosis_label=resolved,
        diagnosis_confidence=diagnosis.confidence,
        used_probe=probe_triggered,
        probe_overrode_diagnosis=resolved is not diagnosis.final_label,
        sample_id=sample_id,
    )


def decide_from_probed(probed: ProbedDiagnosis, sample_id: str | None = None) -> ControllerDecision:
    """Use the Diagnosis and raw Probe result; the Probe's typed final_label is ignored."""
    return decide_action(probed.diagnosis, probed.probe_triggered, probed.probe_result, sample_id)


def run_controller(
    item: DiagnosisInput,
    classifier: HierarchicalDiagnosisClassifier,
    probe: EvidenceProbe,
    trigger: TriggerPolicy = V2_EXPANDED_TRIGGER,
) -> ControllerDecision:
    """Diagnosis -> optional Probe -> Safe Binary resolution -> routing."""
    diagnosis = classifier.diagnose(item)
    probed = apply_probe(item, diagnosis, probe, trigger)
    return decide_from_probed(probed, sample_id=item.sample_id)
