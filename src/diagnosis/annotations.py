"""Read normalized Diagnosis annotation JSONL.

Annotations are offline targets and metadata only. Development code reads the
normalized JSONL written by scripts/normalize_diagnosis_annotations.py, never
the source Excel files. Records must contain exactly ANNOTATION_FIELDS, so gold
or evaluation fields cannot slip in.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from src.diagnosis.labels import DiagnosisLabel, parse_label
from src.diagnosis.schema import DatasetName, normalize_sample_id, parse_dataset

SECONDARY_LABELS: frozenset[str] = frozenset(
    {
        "oracle_recovered_7b",
        "likely_small_model_capability_limitation",
        "unresolved_hard_case",
    }
)

REVIEW_REASONS: frozenset[str] = frozenset(
    {
        "gold_conflict",
        "ambiguous_scope",
        "ai_flagged",
        "label_boundary",
        "claim_wording",
        "oracle_insufficient",
        "context_conflict",
        "output_inconsistent",
    }
)


@dataclass(frozen=True)
class AnnotationRecord:
    sample_id: str
    dataset: DatasetName
    primary_label: DiagnosisLabel
    secondary_label: str | None
    needs_manual_review: bool
    review_reason: str | None
    annotation_source: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "sample_id", normalize_sample_id(self.sample_id))
        object.__setattr__(self, "dataset", parse_dataset(self.dataset))
        object.__setattr__(self, "primary_label", parse_label(self.primary_label))
        if self.secondary_label is not None:
            if self.secondary_label not in SECONDARY_LABELS:
                raise ValueError(f"Invalid secondary_label {self.secondary_label!r}")
            if self.primary_label is not DiagnosisLabel.EVIDENCE_UTILISATION:
                raise ValueError("secondary_label is only allowed for evidence_utilisation")
        if not isinstance(self.needs_manual_review, bool):
            raise ValueError("needs_manual_review must be bool")
        if self.needs_manual_review != (self.review_reason is not None):
            raise ValueError("review_reason must be set if and only if needs_manual_review is true")
        if self.review_reason is not None and self.review_reason not in REVIEW_REASONS:
            raise ValueError(f"Invalid review_reason {self.review_reason!r}")
        if not isinstance(self.annotation_source, str) or not self.annotation_source:
            raise ValueError("annotation_source must be a non-empty string")

    def to_dict(self) -> dict:
        return asdict(self) | {
            "dataset": self.dataset.value,
            "primary_label": self.primary_label.value,
        }


ANNOTATION_FIELDS: tuple[str, ...] = tuple(AnnotationRecord.__dataclass_fields__)


def parse_annotation(raw: Mapping) -> AnnotationRecord:
    keys = set(raw)
    if keys != set(ANNOTATION_FIELDS):
        raise ValueError(
            f"Annotation fields must be exactly {sorted(ANNOTATION_FIELDS)}; "
            f"missing {sorted(set(ANNOTATION_FIELDS) - keys)}, unexpected {sorted(keys - set(ANNOTATION_FIELDS))}"
        )
    return AnnotationRecord(**raw)


def load_annotations(path: Path) -> dict[str, AnnotationRecord]:
    annotations: dict[str, AnnotationRecord] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = parse_annotation(json.loads(line))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            if record.sample_id in annotations:
                raise ValueError(f"{path}:{line_number}: duplicate sample_id {record.sample_id!r}")
            annotations[record.sample_id] = record
    return annotations
