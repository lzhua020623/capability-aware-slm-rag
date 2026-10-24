"""Controller v1: Diagnosis -> optional Probe / frozen selection -> routing.

Routing only: the returned action names a later compensation module, but no
retriever, reranker, generator or model is called here. All functions are
deterministic and read only runtime-visible Diagnosis / Probe / Selection
outputs.

Safe Binary resolution (development policy frozen on the v2-expanded
validation split): the Probe may only turn a Diagnosis failure back into
correct, and only when the Probe says correct, Stage 1 P(failure) <=
SAFE_OVERRIDE_MAX_STAGE1 and max Stage 2 score <= SAFE_OVERRIDE_MAX_STAGE2.
The Probe never changes one failure type into another.

Frozen production-like path (P60/P61/P62): Probe OFF; SelectionDecision from
FrozenOnlineSelectionAdapter; decide()/decision_from_selection() map to
ControllerDecision without calling Enhancement implementations.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from src.controller.failure_representation import (
    representation_from_diagnosis,
    representation_from_primary_only,
)
from src.controller.normalization import normalize_failure_type
from src.controller.schema import (
    ACTION_CONCEPTUAL_ALIASES,
    ControllerAction,
    ControllerDecision,
    SelectionDecision,
    SelectionLane,
)
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
    # FR6 from pre-probe diagnosis scores; if probe flips to correct, clear actives.
    fr6 = representation_from_diagnosis(diagnosis)
    if resolved is DiagnosisLabel.CORRECT:
        primary = DiagnosisLabel.CORRECT
        active: tuple[DiagnosisLabel, ...] = ()
        scores = fr6.failure_scores
    else:
        primary = resolved
        active = fr6.active_failures
        scores = fr6.failure_scores
    return ControllerDecision(
        action=route_controller(resolved),
        diagnosis_label=resolved,
        diagnosis_confidence=diagnosis.confidence,
        used_probe=probe_triggered,
        probe_overrode_diagnosis=resolved is not diagnosis.final_label,
        sample_id=sample_id,
        selection_score=None,
        selection_lane=None,
        reason="legacy_probe_or_diagnosis_routing",
        primary_failure_type=primary,
        active_failures=active,
        failure_scores=scores,
    )


def decide_from_probed(probed: ProbedDiagnosis, sample_id: str | None = None) -> ControllerDecision:
    """Use the Diagnosis and raw Probe result; the Probe's typed final_label is ignored."""
    return decide_action(probed.diagnosis, probed.probe_triggered, probed.probe_result, sample_id)


def decide(
    selection: SelectionDecision,
    *,
    failure_type_override: DiagnosisLabel | str | None = None,
) -> ControllerDecision:
    """Public Controller entry: SelectionDecision -> ControllerDecision.

    Alias of decision_from_selection. Does not call any Enhancement module.
    selection_lane never overrides failure_type. Probe is unused (used_probe=False).
    """
    return decision_from_selection(selection, failure_type_override=failure_type_override)


def decision_from_selection(
    selection: SelectionDecision,
    *,
    failure_type_override: DiagnosisLabel | str | None = None,
) -> ControllerDecision:
    """Map frozen selection to a routing decision without calling Enhancement modules.

    Rules (P62):
      - needs_enhancement=False -> always no_action (any Diagnosis subtype)
      - needs_enhancement=True  -> route by normalized failure_type only
      - unknown/invalid failure_type when selected -> no_action + reason
      - selection_lane is explanatory only (never overrides failure_type)
      - used_probe=False; preserves_initial_answer iff action=no_action
    """
    if not isinstance(selection, SelectionDecision):
        raise TypeError(f"selection must be SelectionDecision, got {type(selection).__name__}")
    if selection.used_probe:
        raise ValueError("frozen selection path requires used_probe=False")

    raw_type = selection.failure_type if failure_type_override is None else failure_type_override
    label = normalize_failure_type(raw_type)
    # FR6 payload from selection (already computed from Diagnosis scores).
    fr6_primary = selection.primary_failure_type or selection.failure_type
    fr6_active = selection.active_failures
    fr6_scores = selection.failure_scores

    if not selection.needs_enhancement:
        out_label = label if label is not None else DiagnosisLabel.CORRECT
        # Representation may still expose multi-failure scores when Diagnosis was failure
        # but selection gated no_action; clear actives only for correct primary.
        if out_label is DiagnosisLabel.CORRECT:
            fr6_active = ()
            fr6_primary = DiagnosisLabel.CORRECT
        return ControllerDecision(
            action=ControllerAction.NO_ACTION,
            diagnosis_label=out_label,
            diagnosis_confidence=selection.confidence,
            used_probe=False,
            probe_overrode_diagnosis=False,
            sample_id=selection.sample_id,
            selection_score=selection.selection_score,
            selection_lane=selection.selection_lane,
            reason="not_selected",
            primary_failure_type=fr6_primary if out_label is not DiagnosisLabel.CORRECT else DiagnosisLabel.CORRECT,
            active_failures=fr6_active if out_label is not DiagnosisLabel.CORRECT else (),
            failure_scores=fr6_scores,
        )

    if label is None:
        return ControllerDecision(
            action=ControllerAction.NO_ACTION,
            diagnosis_label=DiagnosisLabel.CORRECT,
            diagnosis_confidence=selection.confidence,
            used_probe=False,
            probe_overrode_diagnosis=False,
            sample_id=selection.sample_id,
            selection_score=selection.selection_score,
            selection_lane=selection.selection_lane,
            reason=f"unknown_failure_type:{raw_type!r}",
            primary_failure_type=DiagnosisLabel.CORRECT,
            active_failures=(),
            failure_scores=fr6_scores,
        )

    if label is DiagnosisLabel.CORRECT:
        return ControllerDecision(
            action=ControllerAction.NO_ACTION,
            diagnosis_label=DiagnosisLabel.CORRECT,
            diagnosis_confidence=selection.confidence,
            used_probe=False,
            probe_overrode_diagnosis=False,
            sample_id=selection.sample_id,
            selection_score=selection.selection_score,
            selection_lane=selection.selection_lane,
            reason="selected_but_failure_type_correct",
            primary_failure_type=DiagnosisLabel.CORRECT,
            active_failures=(),
            failure_scores=fr6_scores,
        )

    # Single primary route unchanged: route by primary_failure_type / failure_type.
    action = ACTION_BY_LABEL[label]
    return ControllerDecision(
        action=action,
        diagnosis_label=label,
        diagnosis_confidence=selection.confidence,
        used_probe=False,
        probe_overrode_diagnosis=False,
        sample_id=selection.sample_id,
        selection_score=selection.selection_score,
        selection_lane=selection.selection_lane,
        reason="routed_by_failure_type",
        primary_failure_type=label,
        active_failures=fr6_active,
        failure_scores=fr6_scores,
    )


def decide_from_raw_selection(
    *,
    needs_enhancement: bool,
    failure_type: DiagnosisLabel | str | None,
    selection_score: float,
    selection_lane: SelectionLane | str,
    confidence: float,
    p_fail: float = 0.0,
    p_insuff: float = 0.0,
    sample_id: str | None = None,
) -> ControllerDecision:
    """Routing boundary that accepts aliases / unknown types with safe no_action fallback.

    Used for tests and hosts that may pass historical alias strings. Does not
    construct a SelectionDecision when failure_type is unknown.
    """
    lane = SelectionLane(selection_lane)
    label = normalize_failure_type(failure_type)

    if not needs_enhancement:
        return ControllerDecision(
            action=ControllerAction.NO_ACTION,
            diagnosis_label=label if label is not None else DiagnosisLabel.CORRECT,
            diagnosis_confidence=confidence,
            used_probe=False,
            probe_overrode_diagnosis=False,
            sample_id=sample_id,
            selection_score=selection_score,
            selection_lane=SelectionLane.NO_ACTION if lane is SelectionLane.NO_ACTION else lane,
            reason="not_selected",
        )

    if label is None:
        return ControllerDecision(
            action=ControllerAction.NO_ACTION,
            diagnosis_label=DiagnosisLabel.CORRECT,
            diagnosis_confidence=confidence,
            used_probe=False,
            probe_overrode_diagnosis=False,
            sample_id=sample_id,
            selection_score=selection_score,
            selection_lane=lane,
            reason=f"unknown_failure_type:{failure_type!r}",
        )

    if label is DiagnosisLabel.CORRECT:
        return ControllerDecision(
            action=ControllerAction.NO_ACTION,
            diagnosis_label=DiagnosisLabel.CORRECT,
            diagnosis_confidence=confidence,
            used_probe=False,
            probe_overrode_diagnosis=False,
            sample_id=sample_id,
            selection_score=selection_score,
            selection_lane=lane,
            reason="selected_but_failure_type_correct",
        )

    # Build a valid SelectionDecision then route (lane must match needs_enhancement).
    if lane is SelectionLane.NO_ACTION:
        lane = SelectionLane.RESIDUAL_COMBO
    fr6 = representation_from_primary_only(label)
    selection = SelectionDecision(
        needs_enhancement=True,
        failure_type=label,
        selection_score=selection_score,
        selection_lane=lane,
        confidence=confidence,
        p_fail=p_fail,
        p_insuff=p_insuff,
        used_probe=False,
        sample_id=sample_id,
        primary_failure_type=fr6.primary_failure_type,
        active_failures=fr6.active_failures,
        failure_scores=fr6.failure_scores,
    )
    return decision_from_selection(selection)


def run_controller(
    item: DiagnosisInput,
    classifier: HierarchicalDiagnosisClassifier,
    probe: EvidenceProbe | None = None,
    trigger: TriggerPolicy = V2_EXPANDED_TRIGGER,
    *,
    selection_adapter: object | None = None,
    adapter_features: object | None = None,
) -> ControllerDecision:
    """Diagnosis -> optional frozen selection or optional Probe -> routing.

    Frozen adapter path (preferred operational path, Probe OFF):
      pass selection_adapter + adapter_features; probe may be None and is ignored.

    Legacy Probe path:
      pass an EvidenceProbe; selection_adapter must be None.

    Probe-off without selection:
      probe=None and selection_adapter=None keeps Diagnosis routing with no Safe Binary.
    """
    diagnosis = classifier.diagnose(item)
    if selection_adapter is not None:
        if adapter_features is None:
            raise ValueError("adapter_features is required when selection_adapter is set")
        from src.controller.selection import FrozenOnlineSelectionAdapter

        if not isinstance(selection_adapter, FrozenOnlineSelectionAdapter):
            raise TypeError(
                "selection_adapter must be FrozenOnlineSelectionAdapter, "
                f"got {type(selection_adapter).__name__}"
            )
        selection = selection_adapter.select(
            diagnosis, adapter_features, sample_id=item.sample_id
        )
        return decide(selection)
    if probe is None:
        return decide_action(diagnosis, False, None, sample_id=item.sample_id)
    probed = apply_probe(item, diagnosis, probe, trigger)
    return decide_from_probed(probed, sample_id=item.sample_id)


# Re-export for handoff docs / tests
__all_action_aliases__ = ACTION_CONCEPTUAL_ALIASES
