"""Strict one-to-one join of C1 records and normalized annotations.

A development record holds the runtime DiagnosisInput, the offline
target_label and offline metadata. Classifiers may read only `input`;
`metadata` is for offline filtering and analysis and must never be a feature.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from src.diagnosis.adapter import c1_record_to_input
from src.diagnosis.annotations import AnnotationRecord
from src.diagnosis.labels import DiagnosisLabel
from src.diagnosis.schema import DiagnosisInput

MAX_REPORTED_IDS = 5


@dataclass(frozen=True)
class DevelopmentMetadata:
    secondary_label: str | None
    needs_manual_review: bool
    review_reason: str | None
    annotation_source: str


@dataclass(frozen=True)
class DiagnosisDevelopmentRecord:
    input: DiagnosisInput
    target_label: DiagnosisLabel
    metadata: DevelopmentMetadata

    def to_dict(self) -> dict:
        return {
            "input": {
                "sample_id": self.input.sample_id,
                "dataset": self.input.dataset.value,
                "query": self.input.query,
                "evidence": [asdict(item) for item in self.input.evidence],
                "initial_answer": self.input.initial_answer,
            },
            "target_label": self.target_label.value,
            "metadata": asdict(self.metadata),
        }


def load_jsonl(path: Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _preview(ids: Iterable[str]) -> list[str]:
    return sorted(ids)[:MAX_REPORTED_IDS]


def join_development_records(
    c1_records: Iterable[Mapping],
    annotations: Mapping[str, AnnotationRecord],
) -> list[DiagnosisDevelopmentRecord]:
    """Join in C1 order; any missing, extra or duplicate ID raises ValueError."""
    inputs = [c1_record_to_input(record) for record in c1_records]
    input_ids = [item.sample_id for item in inputs]
    duplicates = {sample_id for sample_id, count in Counter(input_ids).items() if count > 1}
    if duplicates:
        raise ValueError(f"Duplicate C1 sample_id values: {_preview(duplicates)}")

    missing = set(input_ids) - set(annotations)
    if missing:
        raise ValueError(f"{len(missing)} C1 records have no annotation, e.g. {_preview(missing)}")
    extra = set(annotations) - set(input_ids)
    if extra:
        raise ValueError(f"{len(extra)} annotations have no C1 record, e.g. {_preview(extra)}")

    records = []
    for item in inputs:
        annotation = annotations[item.sample_id]
        if annotation.sample_id != item.sample_id:
            raise ValueError(f"Annotation keyed {item.sample_id!r} has sample_id {annotation.sample_id!r}")
        if annotation.dataset is not item.dataset:
            raise ValueError(
                f"{item.sample_id}: annotation dataset {annotation.dataset.value!r} "
                f"does not match C1 dataset {item.dataset.value!r}"
            )
        records.append(
            DiagnosisDevelopmentRecord(
                input=item,
                target_label=annotation.primary_label,
                metadata=DevelopmentMetadata(
                    secondary_label=annotation.secondary_label,
                    needs_manual_review=annotation.needs_manual_review,
                    review_reason=annotation.review_reason,
                    annotation_source=annotation.annotation_source,
                ),
            )
        )
    return records
