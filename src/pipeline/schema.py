"""Adaptive RAG host pipeline result schema (P64).

Internal selection feature vectors are intentionally excluded from the public
result. Gold / offline evaluation fields must never appear here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping

from src.compensation.schema import EnhancementResult
from src.controller.schema import ControllerDecision, SelectionDecision
from src.diagnosis.hierarchical_classifier import HierarchicalDiagnosis
from src.diagnosis.schema import EvidenceItem


class PipelineStatus(StrEnum):
    SUCCESS = "success"
    NO_ACTION = "no_action"
    ENHANCEMENT_UNAVAILABLE = "enhancement_unavailable"
    ENHANCEMENT_ERROR = "enhancement_error"


@dataclass(frozen=True)
class AdaptiveRAGResult:
    """End-to-end host output after Controller + EnhancementExecutor."""

    query: str
    initial_answer: str
    final_answer: str
    initial_evidence: tuple[EvidenceItem, ...]
    final_evidence: tuple[EvidenceItem, ...]
    diagnosis: HierarchicalDiagnosis
    selection_decision: SelectionDecision
    controller_decision: ControllerDecision
    enhancement_result: EnhancementResult
    changed_answer: bool
    status: PipelineStatus
    sample_id: str | None = field(default=None, compare=False)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    latency_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "query": self.query,
            "initial_answer": self.initial_answer,
            "final_answer": self.final_answer,
            "initial_evidence": [
                {
                    "rank": e.rank,
                    "passage_id": e.passage_id,
                    "text": e.text,
                    "score": e.score,
                }
                for e in self.initial_evidence
            ],
            "final_evidence": [
                {
                    "rank": e.rank,
                    "passage_id": e.passage_id,
                    "text": e.text,
                    "score": e.score,
                }
                for e in self.final_evidence
            ],
            "diagnosis": {
                "final_label": self.diagnosis.final_label.value,
                "confidence": self.diagnosis.confidence,
                "stage1_failure_probability": self.diagnosis.stage1_failure_probability,
                "stage1_label": self.diagnosis.stage1_label,
            },
            "selection_decision": self.selection_decision.to_dict(),
            "controller_decision": self.controller_decision.to_dict(),
            "enhancement_result": self.enhancement_result.to_dict(),
            "changed_answer": self.changed_answer,
            "status": self.status.value,
            "metadata": dict(self.metadata),
            "latency_ms": self.latency_ms,
        }
