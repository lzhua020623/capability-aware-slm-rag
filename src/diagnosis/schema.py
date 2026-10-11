"""Runtime input/output schema for Diagnosis.

Runtime inputs carry only what is observable at inference time: the query,
the retrieved evidence, and the initial Qwen-A answer. Gold answers/labels,
gold evidence, correctness, EM/F1, C0/C3 outcomes and larger-model outcomes
are offline-only and must never be added here.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

from src.diagnosis.labels import DiagnosisLabel, parse_label


class DatasetName(StrEnum):
    """Canonical dataset names. Source aliases such as natural_questions are mapped by adapters."""

    NQ = "nq"
    FEVER = "fever"

FORBIDDEN_RUNTIME_FIELDS: frozenset[str] = frozenset(
    {
        "gold_answer",
        "gold_label",
        "gold_answer_or_label",
        "gold_evidence",
        "gold_evidence_text",
        "gold_passage_id",
        "gold_page_ids",
        "gold_evidence_in_top5",
        "correct",
        "em",
        "f1",
        "c0_result",
        "c3_result",
        "larger_model_result",
    }
)


def normalize_sample_id(value: str | int) -> str:
    """Diagnosis uses string IDs; FEVER result files store integer IDs."""
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise TypeError(f"sample_id must be str or int, got {type(value).__name__}")
    text = str(value).strip()
    if not text:
        raise ValueError("sample_id must not be empty")
    return text


def parse_dataset(value: DatasetName | str) -> DatasetName:
    if isinstance(value, DatasetName):
        return value
    if isinstance(value, str):
        try:
            return DatasetName(value)
        except ValueError:
            pass
    allowed = ", ".join(name.value for name in DatasetName)
    raise ValueError(f"Invalid dataset {value!r}; expected one of: {allowed}")


def _require_unit_interval(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    value = float(value)
    if math.isnan(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0.0, 1.0], got {value}")
    return value


@dataclass(frozen=True)
class EvidenceItem:
    rank: int
    passage_id: str
    text: str
    score: float

    def __post_init__(self) -> None:
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank < 1:
            raise ValueError(f"rank must be a positive int, got {self.rank!r}")
        if not isinstance(self.passage_id, str) or not self.passage_id:
            raise ValueError("passage_id must be a non-empty string")
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")
        if isinstance(self.score, bool) or not isinstance(self.score, (int, float)):
            raise TypeError(f"score must be a number, got {type(self.score).__name__}")
        if math.isnan(self.score):
            raise ValueError("score must not be NaN")
        object.__setattr__(self, "score", float(self.score))


@dataclass(frozen=True)
class DiagnosisInput:
    sample_id: str
    dataset: DatasetName
    query: str
    evidence: list[EvidenceItem]
    initial_answer: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "sample_id", normalize_sample_id(self.sample_id))
        object.__setattr__(self, "dataset", parse_dataset(self.dataset))
        if not isinstance(self.query, str):
            raise TypeError("query must be a string")
        if not isinstance(self.initial_answer, str):
            raise TypeError("initial_answer must be a string")
        evidence = list(self.evidence)
        if not all(isinstance(item, EvidenceItem) for item in evidence):
            raise TypeError("evidence must contain only EvidenceItem objects")
        ranks = [item.rank for item in evidence]
        if len(set(ranks)) != len(ranks):
            raise ValueError(f"evidence ranks must be unique, got {ranks}")
        object.__setattr__(self, "evidence", evidence)


@dataclass(frozen=True)
class DiagnosisOutput:
    """Diagnosis decision passed to the Controller.

    class_scores is optional debug/evaluation output; the Controller must
    rely only on diagnosis_label, confidence and needs_probe.
    """

    diagnosis_label: DiagnosisLabel
    confidence: float
    needs_probe: bool
    class_scores: Mapping[DiagnosisLabel, float] | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "diagnosis_label", parse_label(self.diagnosis_label))
        object.__setattr__(self, "confidence", _require_unit_interval("confidence", self.confidence))
        if not isinstance(self.needs_probe, bool):
            raise TypeError(f"needs_probe must be bool, got {type(self.needs_probe).__name__}")
        if self.class_scores is not None:
            scores = {
                parse_label(label): _require_unit_interval(f"class_scores[{label}]", score)
                for label, score in self.class_scores.items()
            }
            object.__setattr__(self, "class_scores", scores)
