"""Shared Enhancement contracts with legacy refinement imports preserved.

Refinement dataclasses have one definition in refinement_schema. Their old
import names remain available here for existing Controller callers.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping

from src.controller.schema import ControllerAction, ControllerDecision
from src.diagnosis.labels import DiagnosisLabel, parse_label
from src.diagnosis.schema import EvidenceItem, normalize_sample_id
from src.compensation.refinement_schema import (
    DEFAULT_KEEP_K, DEFAULT_RERANKER_MODEL, DEFAULT_REFINEMENT_CONFIG,
    REFUSAL_ANSWERS, RefinementConfig, RefinementResult, RefinementStatus,
    assert_no_forbidden_fields,
)


class EnhancementStatus(StrEnum):
    """Module execution status — not answer correctness."""

    SUCCESS = "success"
    NO_ACTION = "no_action"
    UNAVAILABLE = "unavailable"
    ERROR = "error"


@dataclass(frozen=True)
class EnhancementRequest:
    """Minimal runtime-safe input shared by all Enhancement modules."""

    query: str
    initial_answer: str
    evidence: tuple[EvidenceItem, ...]
    controller_decision: ControllerDecision
    failure_type: DiagnosisLabel
    sample_id: str | None = field(default=None, compare=False)
    dataset: str | None = None
    context: Mapping[str, Any] | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.query, str):
            raise TypeError("query must be a string")
        if not isinstance(self.initial_answer, str):
            raise TypeError("initial_answer must be a string")
        if not isinstance(self.controller_decision, ControllerDecision):
            raise TypeError("controller_decision must be a ControllerDecision")
        object.__setattr__(self, "failure_type", parse_label(self.failure_type))
        evidence = tuple(self.evidence)
        if not all(isinstance(item, EvidenceItem) for item in evidence):
            raise TypeError("evidence must contain only EvidenceItem objects")
        object.__setattr__(self, "evidence", evidence)
        if self.sample_id is not None:
            object.__setattr__(self, "sample_id", normalize_sample_id(self.sample_id))
        if self.dataset is not None and not isinstance(self.dataset, str):
            raise TypeError("dataset must be str or None")
        if self.context is not None and not isinstance(self.context, Mapping):
            raise TypeError("context must be a mapping or None")
        # Reject common gold/offline keys if someone stuffs them into context.
        if self.context is not None:
            forbidden = {
                "gold_answer",
                "gold_label",
                "gold_evidence",
                "gold_passage_id",
                "target_label",
                "offline_subtype",
                "c_em",
                "e_em",
                "c_answer",
                "e_answer",
            }
            bad = forbidden & set(self.context)
            if bad:
                raise ValueError(f"EnhancementRequest.context contains forbidden keys: {sorted(bad)}")

    @classmethod
    def from_runtime(
        cls,
        *,
        query: str,
        initial_answer: str,
        evidence: list[EvidenceItem] | tuple[EvidenceItem, ...],
        controller_decision: ControllerDecision,
        sample_id: str | None = None,
        dataset: str | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> EnhancementRequest:
        """Build a request; failure_type is taken from ControllerDecision."""
        return cls(
            query=query,
            initial_answer=initial_answer,
            evidence=tuple(evidence),
            controller_decision=controller_decision,
            failure_type=controller_decision.failure_type,
            sample_id=sample_id if sample_id is not None else controller_decision.sample_id,
            dataset=dataset,
            context=context,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "dataset": self.dataset,
            "query": self.query,
            "initial_answer": self.initial_answer,
            "evidence": [
                {
                    "rank": item.rank,
                    "passage_id": item.passage_id,
                    "text": item.text,
                    "score": item.score,
                }
                for item in self.evidence
            ],
            "controller_decision": self.controller_decision.to_dict(),
            "failure_type": self.failure_type.value,
            "context": None if self.context is None else dict(self.context),
        }


@dataclass(frozen=True)
class EnhancementResult:
    """Unified Enhancement output.

    status=success means the module ran without failure — not that the answer is correct.
    """

    action: ControllerAction
    final_answer: str
    evidence: tuple[EvidenceItem, ...]
    changed_answer: bool
    status: EnhancementStatus
    metadata: Mapping[str, Any] = field(default_factory=dict)
    latency_ms: float | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "action", ControllerAction(self.action))
        object.__setattr__(self, "status", EnhancementStatus(self.status))
        if not isinstance(self.final_answer, str):
            raise TypeError("final_answer must be a string")
        if not isinstance(self.changed_answer, bool):
            raise TypeError("changed_answer must be bool")
        evidence = tuple(self.evidence)
        if not all(isinstance(item, EvidenceItem) for item in evidence):
            raise TypeError("evidence must contain only EvidenceItem objects")
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(self, "metadata", dict(self.metadata))
        if self.latency_ms is not None:
            if isinstance(self.latency_ms, bool) or not isinstance(self.latency_ms, (int, float)):
                raise TypeError("latency_ms must be a number or None")
            if not math.isfinite(self.latency_ms) or self.latency_ms < 0:
                raise ValueError("latency_ms must be a non-negative finite number")
            object.__setattr__(self, "latency_ms", float(self.latency_ms))
        if self.error is not None and not isinstance(self.error, str):
            raise TypeError("error must be str or None")

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "final_answer": self.final_answer,
            "evidence": [
                {
                    "rank": item.rank,
                    "passage_id": item.passage_id,
                    "text": item.text,
                    "score": item.score,
                }
                for item in self.evidence
            ],
            "changed_answer": self.changed_answer,
            "status": self.status.value,
            "metadata": dict(self.metadata),
            "latency_ms": self.latency_ms,
            "error": self.error,
        }
