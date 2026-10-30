"""L2: P3-20 with a deeper, hybrid candidate stage (configs/controller/l2_round_v1.json).

Pure functions; the caller owns the retrievers, the scorer and generation. BGE hits come
first, then BM25-only hits; every passage keeps its rank in each list that holds it. With
20 BGE hits, no BM25 hits and RRF fusion the selection equals select_p3_20.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction

from .p3 import CONTEXT_SIZE, RRF_K, _number, _text

FUSIONS = ("rrf", "ce_only")


def union_candidates(bge_hits: Sequence, bm25_hits: Sequence) -> list[dict]:
    """Union by passage_id: BGE order first, then BM25-only passages in BM25 order."""
    pool, index = [], {}
    for source, hits in (("bge", bge_hits), ("bm25", bm25_hits)):
        for rank, row in enumerate(hits, 1):
            pid, text = _text(row["passage_id"], "passage_id"), _text(row["text"], "text")
            if pid not in index:
                index[pid] = len(pool)
                pool.append({"passage_id": pid, "text": text, "ranks": {}})
            elif pool[index[pid]]["text"] != text:
                raise ValueError(f"conflicting text for passage_id={pid}")
            if source in pool[index[pid]]["ranks"]:
                raise ValueError(f"duplicate passage_id={pid} in the {source} list")
            pool[index[pid]]["ranks"][source] = rank
    return pool


def select_l2(pool: Sequence[dict], scores: Sequence[float], fusion: str) -> list[dict]:
    """Top-5 of the union by the registered fusion; ties by union position."""
    if fusion not in FUSIONS:
        raise ValueError(f"fusion must be one of {FUSIONS}")
    if len(scores) != len(pool) or len(pool) < CONTEXT_SIZE:
        raise ValueError("expected one cross-encoder score per candidate and at least 5 candidates")
    values = [_number(s, "cross-encoder score") for s in scores]
    ce_order = sorted(range(len(pool)), key=lambda i: (-values[i], i))
    if fusion == "ce_only":
        order = ce_order[:CONTEXT_SIZE]
    else:
        ce_ranks = {i: rank for rank, i in enumerate(ce_order, 1)}
        fused = [sum((Fraction(1, RRF_K + r) for r in row["ranks"].values()), Fraction(0))
                 + Fraction(1, RRF_K + ce_ranks[i]) for i, row in enumerate(pool)]
        order = sorted(range(len(pool)), key=lambda i: (-fused[i], i))[:CONTEXT_SIZE]
    return [{"passage_id": pool[i]["passage_id"], "text": pool[i]["text"], "rank": rank}
            for rank, i in enumerate(order, 1)]


def same_context(selected: Sequence, base: Sequence) -> bool:
    """P3-20's no_change rule: the same passages and texts in the same order."""
    signature = lambda rows: [(r["passage_id"], r["text"]) for r in rows]
    return signature(selected) == signature(base)
