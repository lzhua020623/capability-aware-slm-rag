"""Pure-Python contracts compatible with Base Retriever evidence dictionaries."""

from dataclasses import dataclass
from typing import Any, TypedDict


class _BaseEvidence(TypedDict):
    passage_id: str
    text: str
    score: float


class Evidence(_BaseEvidence, total=False):
    """Required Base fields plus optional enhancement metadata.

    ``score`` belongs to the first occurrence, never to RRF. Generated provenance
    entries contain source, original list rank, passage_id, score and dataset
    (when known). Existing provenance dictionaries are preserved as metadata.
    Alias IDs exclude the representative passage_id and are dataset-scoped.
    """

    dataset: str
    rank: int
    fusion_score: float
    provenance: list[dict[str, Any]]
    alias_passage_ids: list[str]


class FusionResult(TypedDict):
    """Candidates and selected evidence; new candidates do not imply sufficiency."""

    evidence: list[Evidence]
    candidates: list[Evidence]
    new_evidence_count: int


@dataclass(frozen=True)
class EnhancementBudget:
    """Per invocation limits; no retries or LLM calls are allowed in this MVP."""

    max_retrieval_calls: int = 2
    top_k_per_query: int = 5

    def __post_init__(self) -> None:
        if type(self.max_retrieval_calls) is not int or not 0 <= self.max_retrieval_calls <= 2:
            raise ValueError("max_retrieval_calls must be an integer between 0 and 2")
        if type(self.top_k_per_query) is not int or not 1 <= self.top_k_per_query <= 5:
            raise ValueError("top_k_per_query must be an integer between 1 and 5")


class EnhancementTrace(TypedDict):
    """Operation trace; latency is wall time in seconds, not model-only time."""

    original_query: str
    rewritten_queries: list[str]
    original_evidence_ids: list[str]
    candidate_ids: list[str]
    new_candidate_ids: list[str]
    selected_ids: list[str]
    retrieval_queries: list[str]
    retrieval_calls: int
    llm_calls: int
    latency: float
    status: str
    errors: list[dict[str, str]]


class EnhancementResult(FusionResult):
    trace: EnhancementTrace
