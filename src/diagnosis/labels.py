"""Frozen runtime Diagnosis taxonomy.

Offline analysis categories from the preliminary experiment (7B oracle
recovered, likely small-model capability limitation, unresolved hard case)
are intentionally not part of this taxonomy.
"""

from __future__ import annotations

from enum import StrEnum


class DiagnosisLabel(StrEnum):
    CORRECT = "correct"
    EVIDENCE_INSUFFICIENCY = "evidence_insufficiency"
    EVIDENCE_ACCESSIBILITY = "evidence_accessibility"
    EVIDENCE_UTILISATION = "evidence_utilisation"


RUNTIME_LABELS: tuple[DiagnosisLabel, ...] = tuple(DiagnosisLabel)


def parse_label(value: DiagnosisLabel | str) -> DiagnosisLabel:
    """Return the matching label or raise ValueError for anything outside the taxonomy."""
    if isinstance(value, DiagnosisLabel):
        return value
    if not isinstance(value, str):
        raise ValueError(f"Diagnosis label must be a string, got {type(value).__name__}")
    try:
        return DiagnosisLabel(value)
    except ValueError:
        allowed = ", ".join(label.value for label in RUNTIME_LABELS)
        raise ValueError(f"Invalid diagnosis label {value!r}; expected one of: {allowed}") from None
