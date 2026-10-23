"""NQ backend using an externally loaded E5FaissRetriever configured for BGE.

No model/FAISS import, download, index construction, or resource ownership. The
Runner must load the matching frozen NQ BGE IndexFlatIP and JSONL metadata before
constructing this adapter, and keep them available until retrieval finishes.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Integral, Real

from .types import Evidence

BGE_MODEL_NAME = "BAAI/bge-base-en-v1.5"
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
# FAISS's public MetricType enum; using the value avoids a runtime FAISS import.
_METRIC_INNER_PRODUCT = 0


def _integer(value, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _single_row(matrix, name: str, *, allow_empty: bool = False):
    """Accept numpy-like iterable matrices without requiring numpy in this module."""
    if isinstance(matrix, (str, bytes, Mapping)):
        raise ValueError(f"{name} must be a single-query matrix")
    try:
        rows = list(matrix)
        if not rows and allow_empty:
            return None
        if len(rows) != 1 or isinstance(rows[0], (str, bytes, Mapping)):
            raise ValueError(f"{name} must contain exactly one query row")
        return list(rows[0])
    except TypeError as exc:
        raise ValueError(f"{name} must be a single-query matrix") from exc


class NQFaissBackend:
    """Wrap a loaded Retriever, preserving FAISS order and raw scores.

    Constructor/resource replacement validates the full ID/text mapping. Repeated
    calls check resource counts and selected rows, avoiding a full corpus scan on
    each query. This checks structural consistency, not index/corpus fingerprints;
    the external Runner must ensure it loaded the matching frozen artifacts.
    Backend errors propagate to RetrievalEnhancer's existing safe fallback.
    """

    def __init__(self, retriever, *, max_top_k: int = 5):
        # Top-20 is opt-in for frozen P3-20; existing callers keep the Top-5 cap.
        if type(max_top_k) is not int or not 1 <= max_top_k <= 20:
            raise ValueError("max_top_k must be an integer between 1 and 20")
        self.max_top_k = max_top_k
        self.retriever = retriever
        self._resource_signature = None
        self._check_resources()

    def _record(self, position: int) -> tuple[str, str]:
        record = self.retriever.metadata[position]
        if not isinstance(record, Mapping):
            raise ValueError(f"metadata[{position}] must be a dictionary")
        # Whitelist reads: never inspect or copy answer/Gold annotation fields.
        passage_id = record.get("passage_id")
        text = record.get("text")
        if not isinstance(passage_id, str) or not passage_id.strip():
            raise ValueError(f"metadata[{position}] has an invalid passage_id")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"metadata[{position}] has invalid text")
        if self.retriever.passage_ids[position] != passage_id:
            raise ValueError(f"passage_ids and metadata disagree at position {position}")
        return passage_id, text

    def _check_resources(self) -> tuple[int, int]:
        retriever = self.retriever
        for name, expected in (("model_name", BGE_MODEL_NAME),
                               ("query_prefix", BGE_QUERY_INSTRUCTION), ("passage_prefix", "")):
            if getattr(retriever, name, None) != expected:
                raise ValueError(f"NQ BGE retriever requires {name}={expected!r}")
        if getattr(retriever, "model", None) is None:
            raise RuntimeError("BGE encoder is not loaded or has been unloaded")
        index = getattr(retriever, "index", None)
        if index is None:
            raise RuntimeError("NQ FAISS index must be loaded by the external Runner")
        ntotal = _integer(getattr(index, "ntotal", None), "index.ntotal")
        dimension = _integer(getattr(index, "d", None), "index.d", minimum=1)
        metric = _integer(getattr(index, "metric_type", None), "index.metric_type")
        if metric != _METRIC_INNER_PRODUCT:
            raise ValueError("NQ FAISS index must use inner-product similarity")
        if getattr(index, "is_trained", None) is not True:
            raise ValueError("NQ FAISS index must be trained/ready")
        for name in ("metadata", "passage_ids"):
            items = getattr(retriever, name, None)
            if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
                raise ValueError(f"retriever.{name} must be an ordered sequence")
            if len(items) != ntotal:
                raise ValueError(f"{name} count={len(items)} does not match index.ntotal={ntotal}")
        signature = (id(index), id(retriever.metadata), id(retriever.passage_ids), ntotal, dimension)
        if signature != self._resource_signature:
            seen = set()
            for position in range(ntotal):
                passage_id, _ = self._record(position)
                if passage_id in seen:
                    raise ValueError(f"Duplicate passage_id in NQ metadata: {passage_id!r}")
                seen.add(passage_id)
            self._resource_signature = signature
        return ntotal, dimension

    def retrieve(self, query: str, *, dataset: str, top_k: int) -> list[Evidence]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a nonempty string")
        if dataset not in ("nq", "natural_questions"):
            raise ValueError("NQFaissBackend supports only nq/natural_questions")
        if type(top_k) is not int or not 1 <= top_k <= self.max_top_k:
            raise ValueError(f"top_k must be an integer between 1 and {self.max_top_k}")
        ntotal, dimension = self._check_resources()
        if ntotal == 0:
            return []
        # Same call and batch size as scripts/run_nq_base_rag.py. Pass raw query:
        # encode_queries adds the frozen BGE instruction exactly once internally.
        embeddings = self.retriever.encode_queries([query], batch_size=16)
        vector = _single_row(embeddings, "query embeddings")
        if len(vector) != dimension:
            raise ValueError("query embedding dimension does not match index.d")
        norm_squared = math.fsum(_number(value, "query embedding") ** 2 for value in vector)
        if not math.isclose(norm_squared, 1.0, rel_tol=1e-4, abs_tol=1e-4):
            raise ValueError("query embeddings must be L2-normalized like Base RAG")
        scores, positions = self.retriever.search(embeddings, top_k=top_k)
        score_row = _single_row(scores, "FAISS scores", allow_empty=True)
        position_row = _single_row(positions, "FAISS indices", allow_empty=True)
        if score_row is None or position_row is None:
            if score_row is None and position_row is None:
                return []
            raise ValueError("FAISS scores/indices query-row counts disagree")
        if len(score_row) != len(position_row) or len(position_row) > top_k:
            raise ValueError("FAISS scores/indices lengths disagree or exceed top_k")
        evidence = []
        for score, raw_position in zip(score_row, position_row):
            position = _integer(raw_position, "FAISS position", minimum=-1)
            if position == -1:
                continue  # FAISS pads insufficient hits with -1 and sentinel scores.
            if position >= ntotal:
                raise ValueError(f"FAISS position {position} is outside metadata/index bounds")
            passage_id, text = self._record(position)
            evidence.append({"passage_id": passage_id, "text": text,
                             "score": _number(score, "retrieval score")})
        return evidence
