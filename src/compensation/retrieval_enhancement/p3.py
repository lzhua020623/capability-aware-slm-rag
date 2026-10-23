"""Frozen NQ P3-20, with injected resources and structural Controller inputs.

No Diagnosis/Controller imports: their frozen schemas live on origin/reranking,
not this branch. Only whitelisted runtime fields are read (never object dumps).
The existing rewrite-based RetrievalEnhancer is retained as a separate legacy
entry point. This default production entry point never rewrites a query.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from fractions import Fraction
from numbers import Real
from time import perf_counter
from typing import Protocol
import math

from src.evaluation.qa import normalize_answer

from .enhancer import RetrieverBackend
from .types import Evidence

RERANKER_MODEL = "BAAI/bge-reranker-large"
RERANKER_REVISION = "55611d7bca2a7133960a6d3b71e083071bbfc312"
CANDIDATE_DEPTH = 20
CONTEXT_SIZE = 5
RRF_K = 60

# Kept byte-identical to the frozen NQ C1 builder in generation/qwen.py.
# Importing that module also imports Torch/Transformers; the parity test instead
# executes just its pure constant/builder AST without importing either package.
_C1_INSTRUCTION = (
    "Answer the question using only the provided context. "
    "Return only the shortest answer span. "
    "Do not provide explanations or full sentences."
)


class PinnedScorer(Protocol):
    """Externally loaded scorer; owner attests weights/config, no lazy loading.

    One call scores all 20 pairs. It may use several internal forward batches;
    scoring_calls in the trace counts API invocations, not forward batches.
    """

    model_name: str
    revision: str

    def __call__(self, query: str, texts: Sequence[str]) -> Sequence[float]: ...


def _field(record, name):
    return record[name] if isinstance(record, Mapping) else getattr(record, name)


def _text(value, name, *, allow_empty=False):
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ValueError(f"{name} must be a {'possibly empty' if allow_empty else 'nonempty'} string")
    return value


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    return float(value)


def _evidence(rows, count):
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or len(rows) != count:
        raise ValueError(f"expected exactly {count} evidence records in retrieval order")
    result = []
    for i, row in enumerate(rows, 1):
        # Do not copy arbitrary metadata, answer fields, provenance or labels.
        result.append({
            "passage_id": _text(_field(row, "passage_id"), "passage_id"),
            "text": _text(_field(row, "text"), "text"),
            "score": _number(_field(row, "score"), "retrieval score"),
            "rank": i,
        })
    if len({r["passage_id"] for r in result}) != count:
        raise ValueError("duplicate passage_id in evidence")
    if any(a["score"] < b["score"] for a, b in zip(result, result[1:])):
        raise ValueError("evidence must be in descending BGE retrieval order")
    return result


def select_p3_20(candidates: Sequence[Evidence], scores: Sequence[float]) -> dict:
    """Pure selection: exactly 20 ordered BGE hits and 20 aligned CE scores.

    Equal weights; CE ties and exact RRF ties use original BGE position. Exact
    rational sums reproduce Phase 5-5/5-6 even for mathematically equal ties.
    No ID/text deduplication: that would change the frozen offline policy.
    Raw retrieval scores remain in score; CE/fusion scores are separate.
    """
    pool = _evidence(candidates, CANDIDATE_DEPTH)
    if not isinstance(scores, Sequence) or isinstance(scores, (str, bytes)) or len(scores) != len(pool):
        raise ValueError("expected exactly 20 aligned cross-encoder scores")
    values = [_number(s, "cross-encoder score") for s in scores]
    ce_order = sorted(range(len(pool)), key=lambda i: (-values[i], i))
    ce_ranks = {i: rank for rank, i in enumerate(ce_order, 1)}
    fused = [Fraction(1, RRF_K + i + 1) + Fraction(1, RRF_K + ce_ranks[i])
             for i in range(len(pool))]
    order = sorted(range(len(pool)), key=lambda i: (-fused[i], i))[:CONTEXT_SIZE]
    details = [{"passage_id": row["passage_id"], "bge_rank": i + 1,
                "cross_encoder_score": values[i], "ce_rank": ce_ranks[i],
                "fusion_score": float(fused[i]), "exact_fraction": str(fused[i])}
               for i, row in enumerate(pool)]
    selected = [{**pool[i], "rank": rank, "fusion_score": float(fused[i])}
                for rank, i in enumerate(order, 1)]
    return {"evidence": selected, "candidates": pool, "scores": details}


def build_c1_prompt(query: str, evidence: Sequence[Evidence]) -> str:
    context = "\n\n".join(f"[{i}] {row['text']}" for i, row in enumerate(evidence, 1))
    return f"{_C1_INSTRUCTION}\n\nContext:\n{context}\n\nQuestion: {query}"


def answer_rejection(answer: str, evidence: Sequence[Evidence]) -> str | None:
    """Require a non-refusal contiguous normalized TOKEN span in one passage.

    Token boundaries prevent e.g. 'York' matching 'Yorkshire'. This grounding
    guard is deliberately stricter than the offline answer-substring metric.
    """
    normalized = normalize_answer(answer)
    refusals = {
        "", "unknown", "none", "na", "no answer", "not specified", "not mentioned",
        "not provided", "unanswerable", "insufficient information", "cannot be determined",
        "cannot determine", "i dont know", "i do not know", "not enough information",
        "i cannot answer", "i cant answer", "unable to answer", "not applicable",
    }
    if normalized in refusals or normalized.startswith((
        "i cannot answer ", "i cant answer ", "i do not know ", "i dont know ",
        "sorry ", "im sorry ", "i am sorry ", "as an ai ",
        "insufficient information ", "not enough information ",
        "i cannot determine ", "i cant determine ", "cannot determine ",
        "i am unable to ", "im unable to ", "unable to answer ",
    )):
        return "refusal_or_empty"
    needle = f" {normalized} "
    if not any(needle in f" {normalize_answer(row['text'])} " for row in evidence):
        return "not_grounded_in_selected_evidence"
    return None


class P320RetrievalEnhancer:
    """Controller-facing NQ enhancement; caller owns every model and resource.

    Accepts frozen dataclass objects or mappings with the documented fields.
    initial_answer is the observable C1 prediction, not a reference answer.
    The decision's diagnosis_label/confidence/Probe details are never read.
    All provider exceptions keep the original answer, with an explicit trace.
    """

    rewriting_enabled = False

    def __init__(self, retriever: RetrieverBackend, scorer: PinnedScorer,
                 generate_fn: Callable[[str], str]):
        if not callable(getattr(retriever, "retrieve", None)):
            raise TypeError("retriever must implement retrieve(query, *, dataset, top_k)")
        if not callable(scorer) or not callable(generate_fn):
            raise TypeError("scorer and generate_fn must be callable")
        if (getattr(scorer, "model_name", None), getattr(scorer, "revision", None)) != (
            RERANKER_MODEL, RERANKER_REVISION
        ):
            raise ValueError("scorer must declare the frozen model_name and full revision")
        self.retriever = retriever
        self.scorer = scorer
        self.generate_fn = generate_fn

    def retrieval_enhancement(self, item, decision) -> str:
        """Bind this method as retrieval_enhancement(item, decision) -> str."""
        return self.run(item, decision)["final_answer"]

    def __call__(self, item, decision) -> str:
        return self.retrieval_enhancement(item, decision)

    def run(self, item, decision) -> dict:
        started = perf_counter()
        action = _text(_field(decision, "action"), "action")
        # Whitelist structural reads; never inspect __dict__, to_dict or labels.
        sample_id = _text(_field(item, "sample_id"), "sample_id")
        dataset = _text(_field(item, "dataset"), "dataset")
        query = _text(_field(item, "query"), "query")
        initial = _text(_field(item, "initial_answer"), "initial_answer", allow_empty=True)
        base = _evidence(_field(item, "evidence"), CONTEXT_SIZE)
        record = {
            "sample_id": sample_id, "dataset": dataset, "query": query,
            "status": "", "initial_answer": initial, "final_answer": initial,
            "generated_answer": None, "original_evidence": base, "candidates": [],
            "evidence": base, "scores": [], "prompt": None, "reject_reason": None,
            "error": None, "retrieval_calls": 0, "scoring_calls": 0,
            "scored_pairs": 0, "generation_calls": 0, "llm_calls": 0,
            "rewriting_enabled": False, "rewritten_queries": [],
            "policy": "P3-20", "rrf_k": RRF_K, "reranker_model": RERANKER_MODEL,
            "reranker_revision": RERANKER_REVISION,
            "latency": {key: 0.0 for key in (
                "retrieval", "scoring", "fusion", "prompt", "generation", "validation", "total")},
        }

        def finish(status):
            record["status"] = status
            record["latency"]["total"] = perf_counter() - started
            return record

        def timed(stage, fn):
            start = perf_counter()
            try:
                return fn()
            finally:
                record["latency"][stage] = perf_counter() - start

        if action != "retrieval_enhancement":
            return finish("not_routed")
        if dataset != "nq":
            record["reject_reason"] = "P3-20 supports NQ only"
            return finish("unsupported_dataset")
        stage = "retrieval"
        try:
            record["retrieval_calls"] = 1

            def retrieve():
                hits = self.retriever.retrieve(query, dataset="nq", top_k=CANDIDATE_DEPTH)
                pool = _evidence(hits, CANDIDATE_DEPTH)
                # Retain both bodies for auditing if an ID/text conflict follows.
                record["candidates"] = pool
                originals = {row["passage_id"]: row["text"] for row in base}
                for row in pool:
                    if row["passage_id"] in originals and originals[row["passage_id"]] != row["text"]:
                        raise ValueError(f"conflicting text for passage_id={row['passage_id']}")
                return pool

            record["candidates"] = timed("retrieval", retrieve)
            stage = "scoring"
            record["scoring_calls"] = 1
            scores = timed("scoring", lambda: list(self.scorer(
                query, [row["text"] for row in record["candidates"]])))
            if len(scores) != CANDIDATE_DEPTH:
                raise ValueError("scorer must return exactly 20 aligned scores")
            scores = [_number(value, "cross-encoder score") for value in scores]
            record["scored_pairs"] = len(scores)
            stage = "fusion"
            outcome = timed("fusion", lambda: select_p3_20(record["candidates"], scores))
            record["scores"] = outcome["scores"]
            record["evidence"] = outcome["evidence"]
            signature = lambda rows: [(r["passage_id"], r["text"]) for r in rows]
            if signature(record["evidence"]) == signature(base):
                return finish("no_change")
            stage = "prompt"
            record["prompt"] = timed("prompt", lambda: build_c1_prompt(query, record["evidence"]))
            stage = "generation"
            record["generation_calls"] = record["llm_calls"] = 1
            answer = timed("generation", lambda: self.generate_fn(record["prompt"]))
            if not isinstance(answer, str):
                raise TypeError("generate_fn must return a string")
            record["generated_answer"] = answer
            stage = "validation"
            record["reject_reason"] = timed("validation", lambda: answer_rejection(answer, record["evidence"]))
            if record["reject_reason"] is not None:
                return finish("kept_original")
            record["final_answer"] = answer.strip()
            return finish("answer_replaced")
        except Exception as exc:
            record["error"] = {"stage": stage, "type": type(exc).__name__, "message": str(exc)}
            return finish(f"{stage}_failed")
