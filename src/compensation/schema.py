"""Evidence refinement (rerank) schema for the `evidence_refinement` action.

This module serves the Controller action that `evidence_accessibility` is
routed to: the needed information is inside the retrieved evidence, but
noise, conflict or ordering stops Qwen-A from using it. The refiner reorders
and prunes the evidence it is given and may re-ask the frozen generator once;
it never changes the Diagnosis, the Probe, the retriever or the frozen prompt.

Everything here is built only from runtime-visible information (the
`DiagnosisInput`, the `ControllerDecision`, the optional `ProbeResult` and the
reranker scores). Gold answers/labels, gold evidence, correctness, EM/F1,
C0/C3 and larger-model outcomes are offline-only and must never appear in
these objects; `FORBIDDEN_RUNTIME_FIELDS` lists the field names that are
banned outright.

`EvidenceItem.score` keeps the original retrieval score so that anything
downstream still reads the field it expects; reranker scores are reported
separately in `RefinementResult.rerank_scores`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum

from src.diagnosis.schema import FORBIDDEN_RUNTIME_FIELDS, EvidenceItem
from src.probe.schema import MAX_EVIDENCE_RANK

# Reorder but keep every candidate. Pruning is opt-in: on the frozen 500 NQ C1
# records it changed no rank-1 outcome and cost 14 samples their only
# answer-bearing passage (results/compensation/rerank_frozen500_offline_keep3.json
# vs _keep5.json). Its real benefit -- a shorter, less noisy prompt -- cannot be
# measured without the generator, so it must be chosen there, not here.
DEFAULT_KEEP_K = MAX_EVIDENCE_RANK
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-large"

# Normalised answers that are treated as "the model declined to answer" and are
# never accepted as a replacement answer.
REFUSAL_ANSWERS: frozenset[str] = frozenset(
    {
        "",
        "unknown",
        "none",
        "na",
        "no answer",
        "not specified",
        "not mentioned",
        "not provided",
        "unanswerable",
        "insufficient information",
        "cannot be determined",
        "cannot determine",
        "i dont know",
    }
)


class RefinementStatus(StrEnum):
    """Outcome of one refinement call.

    NOT_ROUTED
        The Controller action was not `evidence_refinement`; nothing was done.
    NO_CHANGE
        Reranking produced the evidence the generator had already seen, so the
        original answer is kept without spending a generation call.
    KEPT_ORIGINAL
        The generator was re-asked on refined evidence, but the new answer was
        rejected (empty, a refusal, or not grounded in the kept evidence), so
        the original answer is kept.
    ANSWER_REPLACED
        The new answer passed the acceptance check and replaces the original.
    """

    NOT_ROUTED = "not_routed"
    NO_CHANGE = "no_change"
    KEPT_ORIGINAL = "kept_original"
    ANSWER_REPLACED = "answer_replaced"


def _finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        raise ValueError(f"{name} must be finite, got {value}")
    return value


@dataclass(frozen=True)
class RefinementConfig:
    """Refinement policy. Defaults are the module's own setting, not a frozen project setting.

    keep_k
        How many passages to keep after reranking. Capped at
        MAX_EVIDENCE_RANK so the refined evidence stays shaped like the
        frozen Reference RAG Top-5 context. The default keeps all of them:
        reranking alone cannot lose evidence, while pruning can, and the
        offline retrieval-side metrics show no gain from it (see
        DEFAULT_KEEP_K above).
    min_score
        Optional absolute reranker-score floor. A passage below it is dropped
        unless dropping it would leave nothing; `None` disables the floor,
        because cross-encoder logits are not calibrated across models.
    probe_hint_weight
        Score adjustment applied with the Probe's rank hints, as
        `weight * (is_conflicting - is_supporting)`. `supporting_ranks` back
        the *initial* answer, which is the answer we were routed here to
        distrust, so a positive weight promotes the passages that contradict
        it. The default is 0.0 (hints ignored): the Probe is a hint rather
        than ground truth, and about half of the samples routed to this action
        are already correct, so entrenching or inverting the initial answer by
        default would risk harm.
    require_grounded
        Accept a replacement answer only if it is grounded in the kept
        evidence (NQ) or parses to SUPPORTS/REFUTES (FEVER).
    regenerate_when_unchanged
        Re-ask the generator even when reranking changed nothing. Off by
        default to avoid a pointless generation call.
    reranker_model
        Cross-encoder used by `CrossEncoderScorer.from_pretrained`.
    """

    keep_k: int = DEFAULT_KEEP_K
    min_score: float | None = None
    probe_hint_weight: float = 0.0
    require_grounded: bool = True
    regenerate_when_unchanged: bool = False
    reranker_model: str = DEFAULT_RERANKER_MODEL

    def __post_init__(self) -> None:
        if isinstance(self.keep_k, bool) or not isinstance(self.keep_k, int):
            raise TypeError(f"keep_k must be an int, got {type(self.keep_k).__name__}")
        if not 1 <= self.keep_k <= MAX_EVIDENCE_RANK:
            raise ValueError(f"keep_k must be in [1, {MAX_EVIDENCE_RANK}], got {self.keep_k}")
        if self.min_score is not None:
            object.__setattr__(self, "min_score", _finite("min_score", self.min_score))
        object.__setattr__(self, "probe_hint_weight", _finite("probe_hint_weight", self.probe_hint_weight))
        for name in ("require_grounded", "regenerate_when_unchanged"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool")
        if not isinstance(self.reranker_model, str) or not self.reranker_model.strip():
            raise ValueError("reranker_model must be a non-empty string")

    def to_dict(self) -> dict:
        return {
            "keep_k": self.keep_k,
            "min_score": self.min_score,
            "probe_hint_weight": self.probe_hint_weight,
            "require_grounded": self.require_grounded,
            "regenerate_when_unchanged": self.regenerate_when_unchanged,
            "reranker_model": self.reranker_model,
        }


DEFAULT_REFINEMENT_CONFIG = RefinementConfig()


@dataclass(frozen=True)
class RefinementResult:
    """What the refiner did for one sample.

    `final_answer` is what the host RAG should report. `evidence` is the
    refined context in its new rank order (ranks renumbered from 1), with each
    item's original retrieval score preserved; `rerank_scores` is aligned with
    it, or empty when no reranking took place. `sample_id` is for logging only
    and is excluded from equality.
    """

    status: RefinementStatus
    initial_answer: str
    final_answer: str
    evidence: tuple[EvidenceItem, ...]
    rerank_scores: tuple[float, ...]
    original_passage_ids: tuple[str, ...]
    dropped_passage_ids: tuple[str, ...]
    candidate_pool_size: int
    regenerated: bool
    used_probe_hint: bool
    reject_reason: str | None = None
    sample_id: str | None = field(default=None, compare=False)
    answer_changed: bool = field(init=False)
    evidence_changed: bool = field(init=False)
    refined_passage_ids: tuple[str, ...] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", RefinementStatus(self.status))
        for name in ("initial_answer", "final_answer"):
            if not isinstance(getattr(self, name), str):
                raise TypeError(f"{name} must be a string")
        evidence = tuple(self.evidence)
        if not all(isinstance(item, EvidenceItem) for item in evidence):
            raise TypeError("evidence must contain only EvidenceItem objects")
        object.__setattr__(self, "evidence", evidence)
        scores = tuple(_finite("rerank_scores", score) for score in self.rerank_scores)
        if scores and len(scores) != len(evidence):
            raise ValueError(f"rerank_scores has {len(scores)} entries for {len(evidence)} evidence items")
        object.__setattr__(self, "rerank_scores", scores)
        for name in ("original_passage_ids", "dropped_passage_ids"):
            ids = tuple(getattr(self, name))
            if not all(isinstance(value, str) and value for value in ids):
                raise ValueError(f"{name} must contain non-empty strings")
            object.__setattr__(self, name, ids)
        if isinstance(self.candidate_pool_size, bool) or not isinstance(self.candidate_pool_size, int):
            raise TypeError("candidate_pool_size must be an int")
        if self.candidate_pool_size < len(evidence):
            raise ValueError(
                f"candidate_pool_size ({self.candidate_pool_size}) is smaller than the kept evidence ({len(evidence)})"
            )
        for name in ("regenerated", "used_probe_hint"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool")
        if self.reject_reason is not None and not isinstance(self.reject_reason, str):
            raise TypeError("reject_reason must be a string or None")
        refined_ids = tuple(item.passage_id for item in evidence)
        object.__setattr__(self, "refined_passage_ids", refined_ids)
        object.__setattr__(self, "answer_changed", self.final_answer != self.initial_answer)
        object.__setattr__(self, "evidence_changed", refined_ids != self.original_passage_ids)
        self._check_status()

    def _check_status(self) -> None:
        kept_statuses = (RefinementStatus.NOT_ROUTED, RefinementStatus.NO_CHANGE, RefinementStatus.KEPT_ORIGINAL)
        if self.status in kept_statuses and self.answer_changed:
            raise ValueError(f"status {self.status.value!r} must keep the initial answer")
        if self.status is RefinementStatus.NOT_ROUTED:
            if self.regenerated or self.evidence_changed:
                raise ValueError("not_routed must leave the evidence and the answer untouched")
        if self.status is RefinementStatus.NO_CHANGE and self.regenerated:
            raise ValueError("no_change must not spend a generation call")
        if self.status in (RefinementStatus.KEPT_ORIGINAL, RefinementStatus.ANSWER_REPLACED) and not self.regenerated:
            raise ValueError(f"status {self.status.value!r} requires regenerated=True")
        if self.status is RefinementStatus.KEPT_ORIGINAL and self.reject_reason is None:
            raise ValueError("kept_original requires a reject_reason")
        if self.status is RefinementStatus.ANSWER_REPLACED and self.reject_reason is not None:
            raise ValueError("answer_replaced must not carry a reject_reason")

    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "status": self.status.value,
            "initial_answer": self.initial_answer,
            "final_answer": self.final_answer,
            "answer_changed": self.answer_changed,
            "evidence_changed": self.evidence_changed,
            "original_passage_ids": list(self.original_passage_ids),
            "refined_passage_ids": list(self.refined_passage_ids),
            "dropped_passage_ids": list(self.dropped_passage_ids),
            "rerank_scores": list(self.rerank_scores),
            "candidate_pool_size": self.candidate_pool_size,
            "regenerated": self.regenerated,
            "used_probe_hint": self.used_probe_hint,
            "reject_reason": self.reject_reason,
        }


def assert_no_forbidden_fields(names: object) -> None:
    """Raise if any runtime field name is one of the offline-only fields."""
    leaked = sorted(set(names) & FORBIDDEN_RUNTIME_FIELDS)
    if leaked:
        raise ValueError(f"Runtime refinement objects must not expose offline-only fields: {leaked}")


from typing import Any, Mapping
from src.controller.schema import ControllerAction, ControllerDecision
from src.diagnosis.labels import DiagnosisLabel, parse_label
from src.diagnosis.schema import normalize_sample_id

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
