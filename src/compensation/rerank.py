"""Cross-encoder reranking of the retrieved evidence.

Reranking is the mechanical half of the `evidence_refinement` action: score
every candidate passage against the query with a cross-encoder, reorder by
that score, and drop the passages that fall outside `keep_k`. This targets
`evidence_accessibility` directly -- the needed information is present but
buried, out of order, or surrounded by retrieval noise -- without touching the
retriever, the index or the frozen Top-5 retrieval itself.

The scorer is injected (`RelevanceScorer`), so tests never load a model.
`CrossEncoderScorer.from_pretrained` loads a sentence-transformers
CrossEncoder (default `BAAI/bge-reranker-large`).

Candidates default to the evidence the Reference RAG already showed the
generator. A `candidate_provider` may widen the pool (for example a deeper
Top-50 from the frozen index); pool order stays "the original evidence first,
then the extra candidates", and every ordering decision breaks ties on the
candidate's original rank, so the whole function is deterministic.

Probe rank hints are matched on the candidate's original rank and are applied
only when `RefinementConfig.probe_hint_weight` is non-zero.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from src.compensation.refinement_schema import DEFAULT_REFINEMENT_CONFIG, RefinementConfig
from src.diagnosis.schema import DiagnosisInput, EvidenceItem
from src.probe.schema import ProbeResult

CandidateProvider = Callable[[DiagnosisInput], Iterable[EvidenceItem]]


@runtime_checkable
class RelevanceScorer(Protocol):
    """Scores one query against several passage texts. Higher is more relevant."""

    def __call__(self, query: str, texts: Sequence[str]) -> Sequence[float]: ...


class CrossEncoderScorer:
    """Cross-encoder relevance scorer. The prediction function is injected or loaded lazily."""

    def __init__(self, predict_fn: Callable[[list[list[str]]], Sequence[float]], model_name: str | None = None):
        self._predict = predict_fn
        self.model_name = model_name

    @classmethod
    def from_pretrained(cls, model_name: str = DEFAULT_REFINEMENT_CONFIG.reranker_model, **kwargs) -> CrossEncoderScorer:
        """Load a sentence-transformers CrossEncoder. kwargs go to the CrossEncoder constructor."""
        from sentence_transformers import CrossEncoder

        model = CrossEncoder(model_name, **kwargs)

        def predict(pairs: list[list[str]]) -> Sequence[float]:
            return model.predict(pairs)

        return cls(predict, model_name=model_name)

    def __call__(self, query: str, texts: Sequence[str]) -> list[float]:
        if not isinstance(query, str):
            raise TypeError(f"query must be a string, got {type(query).__name__}")
        texts = list(texts)
        if not texts:
            return []
        scores = self._predict([[query, text] for text in texts])
        return [float(score) for score in scores]


class ShuffleScorer:
    """Deterministic pseudo-random scores: a control, not a ranker.

    Reordering evidence changes the prompt whether or not the new order is any
    better, and a re-asked generator sometimes lands on a different answer for
    that reason alone. Scoring a run with this measures how much of a reranker's
    apparent repair rate is that perturbation rather than ranking. Published
    work reports the same concern: on Qwen2.5-7B with BGE Top-5, rearranging
    passages is statistically indistinguishable from shuffling them
    (arXiv:2505.15561).

    Scores hash (seed, query, passage text), so a run reproduces exactly and
    does not depend on the order the candidates arrive in.
    """

    def __init__(self, seed: int = 42):
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError(f"seed must be an int, got {type(seed).__name__}")
        self.seed = seed
        self.model_name = f"shuffle:{seed}"

    def __call__(self, query: str, texts: Sequence[str]) -> list[float]:
        if not isinstance(query, str):
            raise TypeError(f"query must be a string, got {type(query).__name__}")
        scores = []
        for text in texts:
            digest = hashlib.blake2b(repr((self.seed, query, text)).encode("utf-8"), digest_size=8).digest()
            scores.append(int.from_bytes(digest, "big") / 2**64)
        return scores


SHUFFLE_PREFIX = "shuffle:"


def scorer_from_name(name: str) -> RelevanceScorer:
    """Build the scorer a run recorded in `RefinementConfig.reranker_model`.

    `shuffle:<seed>` rebuilds the control; anything else is a cross-encoder
    path or hub id. Offline analyses use this so they reproduce the ordering
    the run actually used instead of assuming a cross-encoder.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("scorer name must be a non-empty string")
    if name.startswith(SHUFFLE_PREFIX):
        seed = name[len(SHUFFLE_PREFIX) :]
        if not seed.lstrip("-").isdigit():
            raise ValueError(f"Invalid shuffle scorer name {name!r}; expected {SHUFFLE_PREFIX}<int seed>")
        return ShuffleScorer(int(seed))
    return CrossEncoderScorer.from_pretrained(name)


@dataclass(frozen=True)
class RerankOutcome:
    """Reranked evidence in its new order, with the scores that produced it.

    `evidence` is renumbered from rank 1 and keeps each passage's original
    retrieval score; `scores` is the ordering score per kept passage,
    including any Probe hint adjustment.
    """

    evidence: tuple[EvidenceItem, ...]
    scores: tuple[float, ...]
    dropped_passage_ids: tuple[str, ...]
    candidate_pool_size: int
    used_probe_hint: bool


def build_candidate_pool(item: DiagnosisInput, candidate_provider: CandidateProvider | None = None) -> tuple[EvidenceItem, ...]:
    """The evidence already shown to the generator, then any extra candidates, deduplicated by passage_id.

    Extra candidates are renumbered after the original ranks so that Probe
    rank hints keep pointing at the passages the Probe actually saw.
    """
    pool = list(item.evidence)
    if candidate_provider is None:
        return tuple(pool)
    seen = {candidate.passage_id for candidate in pool}
    next_rank = max((candidate.rank for candidate in pool), default=0) + 1
    for candidate in candidate_provider(item):
        if not isinstance(candidate, EvidenceItem):
            raise TypeError(f"candidate_provider must yield EvidenceItem objects, got {type(candidate).__name__}")
        if candidate.passage_id in seen:
            continue
        seen.add(candidate.passage_id)
        pool.append(
            EvidenceItem(
                rank=next_rank,
                passage_id=candidate.passage_id,
                text=candidate.text,
                score=candidate.score,
            )
        )
        next_rank += 1
    return tuple(pool)


def _hint_adjustments(
    candidates: Sequence[EvidenceItem],
    probe_result: ProbeResult | None,
    weight: float,
) -> tuple[list[float], bool]:
    """weight * (is_conflicting - is_supporting) per candidate, matched on the original rank."""
    if probe_result is None or weight == 0.0:
        return [0.0] * len(candidates), False
    supporting = set(probe_result.supporting_ranks)
    conflicting = set(probe_result.conflicting_ranks)
    adjustments = [
        weight * ((1.0 if candidate.rank in conflicting else 0.0) - (1.0 if candidate.rank in supporting else 0.0))
        for candidate in candidates
    ]
    return adjustments, any(adjustment != 0.0 for adjustment in adjustments)


def rerank_evidence(
    query: str,
    candidates: Sequence[EvidenceItem],
    scorer: RelevanceScorer,
    config: RefinementConfig = DEFAULT_REFINEMENT_CONFIG,
    probe_result: ProbeResult | None = None,
) -> RerankOutcome:
    """Score, reorder and prune the candidate evidence. Deterministic; ties break on the original rank."""
    candidates = list(candidates)
    if not candidates:
        raise ValueError("cannot rerank an empty candidate pool")
    raw_scores = list(scorer(query, [candidate.text for candidate in candidates]))
    if len(raw_scores) != len(candidates):
        raise ValueError(f"scorer returned {len(raw_scores)} scores for {len(candidates)} candidates")
    adjustments, used_probe_hint = _hint_adjustments(candidates, probe_result, config.probe_hint_weight)
    scores: list[float] = []
    for index, score in enumerate(raw_scores):
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise TypeError(f"scorer returned a non-numeric score at index {index}: {score!r}")
        score = float(score)
        if math.isnan(score):
            raise ValueError(f"scorer returned NaN at index {index}")
        scores.append(score + adjustments[index])

    order = sorted(range(len(candidates)), key=lambda index: (-scores[index], candidates[index].rank))
    kept = order[: config.keep_k]
    if config.min_score is not None:
        above_floor = [index for index in kept if scores[index] >= config.min_score]
        kept = above_floor or kept[:1]

    kept_set = set(kept)
    evidence = tuple(
        EvidenceItem(
            rank=position,
            passage_id=candidates[index].passage_id,
            text=candidates[index].text,
            score=candidates[index].score,
        )
        for position, index in enumerate(kept, start=1)
    )
    return RerankOutcome(
        evidence=evidence,
        scores=tuple(scores[index] for index in kept),
        dropped_passage_ids=tuple(
            candidate.passage_id for index, candidate in enumerate(candidates) if index not in kept_set
        ),
        candidate_pool_size=len(candidates),
        used_probe_hint=used_probe_hint,
    )
