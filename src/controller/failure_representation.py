"""FR6 multi-failure representation from runtime-safe Diagnosis scores.

Uses HierarchicalDiagnosis joint / Stage-2 conditional subtype probabilities only.
No gold, offline labels, new models, or threshold retuning.

Active-failure rule (existing Stage-2 semantics, no new Train search):
  - correct / Stage-1 correct → active_failures = ()
  - otherwise, a failure subtype is active iff its Stage-2 conditional
    probability is >= 1/3 (uniform prior among the three Stage-2 classes).
  - primary_failure_type remains Diagnosis.final_label (Stage-2 argmax).
  - If primary is a failure subtype, it is always included in active_failures.
  - Action priority among simultaneous actives (representation only; does not
    change active set): insufficiency → accessibility → utilisation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from src.diagnosis.hierarchical_classifier import HierarchicalDiagnosis
from src.diagnosis.labels import DiagnosisLabel

FAILURE_SUBTYPES: tuple[DiagnosisLabel, ...] = (
    DiagnosisLabel.EVIDENCE_INSUFFICIENCY,
    DiagnosisLabel.EVIDENCE_ACCESSIBILITY,
    DiagnosisLabel.EVIDENCE_UTILISATION,
)

# Stage-2 is a 3-way softmax; 1/3 is the uniform baseline (not a newly tuned gate).
STAGE2_UNIFORM_PRIOR = 1.0 / 3.0

# Action priority when multiple failures are active (retrieval gap first).
ACTION_PRIORITY: tuple[DiagnosisLabel, ...] = FAILURE_SUBTYPES


@dataclass(frozen=True)
class FailureRepresentation:
    """Multi-failure view for SelectionDecision / ControllerDecision."""

    primary_failure_type: DiagnosisLabel
    active_failures: tuple[DiagnosisLabel, ...]
    failure_scores: Mapping[str, float]

    @property
    def failure_type(self) -> DiagnosisLabel:
        """Backward-compatible alias of primary_failure_type."""
        return self.primary_failure_type


def joint_failure_scores(diagnosis: HierarchicalDiagnosis) -> dict[str, float]:
    """Runtime-safe joint P(insuff/access/util). Missing scores → 0.0."""
    out = {label.value: 0.0 for label in FAILURE_SUBTYPES}
    if diagnosis.class_scores is not None:
        for label in FAILURE_SUBTYPES:
            out[label.value] = float(diagnosis.class_scores.get(label, 0.0))
        return out
    if diagnosis.stage2_class_scores is not None:
        p_fail = float(diagnosis.stage1_failure_probability)
        for label in FAILURE_SUBTYPES:
            cond = float(diagnosis.stage2_class_scores.get(label, 0.0))
            out[label.value] = p_fail * cond
    return out


def stage2_conditionals(diagnosis: HierarchicalDiagnosis) -> dict[DiagnosisLabel, float] | None:
    if diagnosis.stage2_class_scores is not None:
        return {label: float(diagnosis.stage2_class_scores.get(label, 0.0)) for label in FAILURE_SUBTYPES}
    if diagnosis.class_scores is None:
        return None
    p_fail = float(diagnosis.stage1_failure_probability)
    if p_fail <= 0.0:
        return None
    return {
        label: float(diagnosis.class_scores.get(label, 0.0)) / p_fail for label in FAILURE_SUBTYPES
    }


def primary_among_active(active: tuple[DiagnosisLabel, ...]) -> DiagnosisLabel | None:
    """Pick action primary by insuff → access → util; None if empty."""
    if not active:
        return None
    active_set = set(active)
    for label in ACTION_PRIORITY:
        if label in active_set:
            return label
    return active[0]


def representation_from_diagnosis(diagnosis: HierarchicalDiagnosis) -> FailureRepresentation:
    """Build FR6 fields from HierarchicalDiagnosis without changing routing models."""
    scores = joint_failure_scores(diagnosis)
    no_failure = (
        diagnosis.final_label is DiagnosisLabel.CORRECT or diagnosis.stage1_label == "correct"
    )
    if no_failure:
        return FailureRepresentation(
            primary_failure_type=DiagnosisLabel.CORRECT,
            active_failures=(),
            failure_scores=scores,
        )

    primary = diagnosis.final_label
    conditionals = stage2_conditionals(diagnosis)
    active: list[DiagnosisLabel] = []
    if conditionals is not None:
        for label in FAILURE_SUBTYPES:
            if conditionals.get(label, 0.0) >= STAGE2_UNIFORM_PRIOR:
                active.append(label)
    # Primary (Stage-2 argmax) is always active on a failure path.
    if primary in FAILURE_SUBTYPES and primary not in active:
        active.append(primary)
    # Stable order: insuff → access → util (does not change membership).
    active_ordered = tuple(label for label in ACTION_PRIORITY if label in set(active))
    return FailureRepresentation(
        primary_failure_type=primary,
        active_failures=active_ordered,
        failure_scores=scores,
    )


def representation_from_primary_only(primary: DiagnosisLabel) -> FailureRepresentation:
    """Fallback when only a single label is available (raw routing helpers)."""
    if primary is DiagnosisLabel.CORRECT:
        return FailureRepresentation(
            primary_failure_type=DiagnosisLabel.CORRECT,
            active_failures=(),
            failure_scores={label.value: 0.0 for label in FAILURE_SUBTYPES},
        )
    scores = {label.value: 0.0 for label in FAILURE_SUBTYPES}
    scores[primary.value] = 1.0
    return FailureRepresentation(
        primary_failure_type=primary,
        active_failures=(primary,),
        failure_scores=scores,
    )
