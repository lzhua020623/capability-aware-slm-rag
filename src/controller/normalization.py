"""Centralized failure-type normalization for Controller routing.

All historical aliases must be resolved here. Callers must not scatter
string comparisons across modules. Unknown values return None so the
Controller can safely fall back to no_action.
"""

from __future__ import annotations

from src.diagnosis.labels import DiagnosisLabel

# Explicit alias table (lowercase, stripped). Canonical values map to themselves.
FAILURE_TYPE_ALIASES: dict[str, DiagnosisLabel] = {
    # canonical
    "correct": DiagnosisLabel.CORRECT,
    "evidence_insufficiency": DiagnosisLabel.EVIDENCE_INSUFFICIENCY,
    "evidence_accessibility": DiagnosisLabel.EVIDENCE_ACCESSIBILITY,
    "evidence_utilisation": DiagnosisLabel.EVIDENCE_UTILISATION,
    # short / historical aliases
    "insuff": DiagnosisLabel.EVIDENCE_INSUFFICIENCY,
    "insufficiency": DiagnosisLabel.EVIDENCE_INSUFFICIENCY,
    "evidence_insufficient": DiagnosisLabel.EVIDENCE_INSUFFICIENCY,
    "access": DiagnosisLabel.EVIDENCE_ACCESSIBILITY,
    "accessibility": DiagnosisLabel.EVIDENCE_ACCESSIBILITY,
    "util": DiagnosisLabel.EVIDENCE_UTILISATION,
    "utilisation": DiagnosisLabel.EVIDENCE_UTILISATION,
    "utilization": DiagnosisLabel.EVIDENCE_UTILISATION,
    "evidence_utilization": DiagnosisLabel.EVIDENCE_UTILISATION,
}


def normalize_failure_type(value: DiagnosisLabel | str | None) -> DiagnosisLabel | None:
    """Return a DiagnosisLabel, or None if the value is unknown/invalid.

    Never guesses a routing target for unknown inputs.
    """
    if value is None:
        return None
    if isinstance(value, DiagnosisLabel):
        return value
    if not isinstance(value, str):
        return None
    key = value.strip().lower()
    if not key:
        return None
    return FAILURE_TYPE_ALIASES.get(key)
