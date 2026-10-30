"""P3-BM25: P3-20, whose answer is replaced when BM25 brings new evidence (configs/controller/bm25_gate_v1.json).

After P3-20 has run, BM25 top 20 joins P3-20's BGE top 20 (BGE order first, then BM25-only passages);
the GPU cross-encoder scores that union, and its top 5 by score is the L2 context. Only when that
context holds a passage outside the BGE top 20 is the C1 prompt regenerated over it; the answer then
passes answer_rejection or the C1 answer is kept. Otherwise P3-20's record stands unchanged. Errors
in the BM25 step keep P3-20's record, with the error in the trace.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import hashlib
from pathlib import Path
from time import perf_counter

from .p3 import RERANKER_MODEL, RERANKER_REVISION, _number, answer_rejection, build_c1_prompt
from .p3_l2 import select_l2, union_candidates

K_BGE = 20
K_BM25 = 20
FUSION = "ce_only"
SCORER_BATCH = 32
REPLACEABLE = ("no_change", "answer_replaced", "kept_original")


@dataclass(frozen=True)
class P3Bm25Resources:
    """index.search(query, k) -> [(corpus position, score)]; record(position) -> (passage_id, text)."""
    index: object
    record: Callable
    scorer: Callable
    receipt: dict


class P3Bm25:
    def __init__(self, search: Callable, record: Callable, scorer: Callable, generate_fn: Callable[[str], str]):
        self.search, self.record, self.scorer, self.generate_fn = search, record, scorer, generate_fn

    def run(self, p3: dict) -> dict:
        """P3-20's record, or a P3-BM25 record that keeps P3-20's under `p3_20`; `bm25` holds the trace."""
        started = perf_counter()
        trace = {"policy": "P3-BM25", "applied": False, "reason": None, "bm25_hits": [], "pool": [],
                 "scores": [], "selected": [], "bm25_only": [], "error": None, "seconds": 0.0}
        keep = lambda reason: {**p3, "bm25": {**trace, "reason": reason, "seconds": perf_counter() - started}}
        if p3.get("dataset") != "nq" or p3.get("status") not in REPLACEABLE:
            return keep("p3_20_not_completed")
        query, initial = p3["query"], p3["initial_answer"]
        bge = p3["candidates"][:K_BGE]
        stage = "bm25"
        try:
            rows = []
            for rank, (position, score) in enumerate(self.search(query, K_BM25), 1):
                passage_id, text = self.record(position)
                rows.append({"passage_id": passage_id, "text": text, "score": float(score), "rank": rank})
            trace["bm25_hits"] = [[r["passage_id"], r["score"]] for r in rows]
            pool = union_candidates(bge, rows)
            trace["pool"] = [[r["passage_id"], r["ranks"]] for r in pool]
            stage = "scoring"
            scores = [_number(s, "cross-encoder score") for s in self.scorer(query, [r["text"] for r in pool])]
            if len(scores) != len(pool):
                raise ValueError("scorer must return one score per pooled passage")
            trace["scores"] = scores
            stage = "fusion"
            selected = [r["passage_id"] for r in select_l2(pool, scores, FUSION)]
            bge_ids = {r["passage_id"] for r in bge}
            trace["selected"] = selected
            trace["bm25_only"] = [pid for pid in selected if pid not in bge_ids]
            if not trace["bm25_only"]:
                return keep("no_bm25_passage")
            source = {r["passage_id"]: r for r in rows} | {r["passage_id"]: r for r in bge}
            evidence = [{"passage_id": pid, "text": source[pid]["text"], "score": source[pid]["score"], "rank": i}
                        for i, pid in enumerate(selected, 1)]
            stage = "generation"
            prompt = build_c1_prompt(query, evidence)
            answer = self.generate_fn(prompt)
            if not isinstance(answer, str):
                raise TypeError("generate_fn must return a string")
            stage = "validation"
            reason = answer_rejection(answer, evidence)
        except Exception as exc:
            trace["error"] = {"stage": stage, "type": type(exc).__name__, "message": str(exc)}
            return keep(f"{stage}_failed")
        trace.update(applied=True, seconds=perf_counter() - started)
        p3_20 = {k: p3[k] for k in ("status", "final_answer", "generated_answer", "reject_reason", "evidence", "prompt")}
        return {**p3, "policy": "P3-BM25", "p3_20": p3_20, "bm25": trace, "evidence": evidence, "prompt": prompt,
                "generated_answer": answer, "reject_reason": reason,
                "final_answer": initial if reason is not None else answer.strip(),
                "status": "kept_original" if reason is not None else "answer_replaced",
                "retrieval_calls": p3["retrieval_calls"] + 1, "scoring_calls": p3["scoring_calls"] + 1,
                "scored_pairs": p3["scored_pairs"] + len(pool), "generation_calls": p3["generation_calls"] + 1,
                "llm_calls": p3["llm_calls"] + 1}


def directory_sha(directory):
    """One hash over every file under `directory`: relative path, then bytes, in sorted path order."""
    digest = hashlib.sha256()
    for path in sorted(p for p in Path(directory).rglob("*") if p.is_file()):
        digest.update(path.relative_to(directory).as_posix().encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def check_spec(spec):
    if (spec["k_bge"], spec["k_bm25"], spec["fusion"]) != (K_BGE, K_BM25, FUSION):
        raise ValueError("P3-BM25 is registered for BGE 20, BM25 20 and ce_only only")
    scorer = spec["scorer"]
    if scorer != {"model": RERANKER_MODEL, "revision": RERANKER_REVISION, "device": "cuda", "dtype": "float32",
                  "tf32": False, "max_length": 512, "batch_size": SCORER_BATCH, "activation": "identity"}:
        raise ValueError("P3-BM25 needs the registered GPU cross-encoder")


def load_resources(spec, backend, root):
    """Checked BM25 index plus the GPU cross-encoder; `backend` is the loaded NQ Train BGE backend."""
    from .bm25 import Bm25Index
    check_spec(spec)
    directory = Path(root) / spec["index_dir"]
    if directory_sha(directory) != spec["index_sha256"]:
        raise ValueError("BM25 index missing or changed: " + spec["index_dir"])
    index = Bm25Index(directory)
    if index.passage_ids != list(backend.retriever.passage_ids):
        raise ValueError("BM25 document order differs from the BGE corpus order")
    return P3Bm25Resources(index=index, record=backend._record, scorer=cuda_scorer(), receipt=index.receipt)


def cuda_scorer():
    """bge-reranker-large on GPU: FP32, raw logits, max_length 512, TF32 off, batch 32."""
    import torch
    from sentence_transformers import CrossEncoder
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    encoder = CrossEncoder(RERANKER_MODEL, revision=RERANKER_REVISION, device="cuda", max_length=512,
                           activation_fn=torch.nn.Identity(), model_kwargs={"torch_dtype": torch.float32})
    return lambda query, texts: [float(s) for s in encoder.predict([(query, t) for t in texts],
                                                                   batch_size=SCORER_BATCH, show_progress_bar=False)]
