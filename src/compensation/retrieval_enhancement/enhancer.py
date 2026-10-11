"""One-round, budgeted enhancement with injected LLM-free providers."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from time import perf_counter
from typing import Protocol

from .fusion import rrf_fuse
from .rewriting import QueryRewriter, RuleQueryRewriter, filter_rewrites
from .types import EnhancementBudget, EnhancementResult, EnhancementTrace, Evidence


class RetrieverBackend(Protocol):
    """One call retrieves one query from the supplied dataset, in rank order."""

    def retrieve(self, query: str, *, dataset: str, top_k: int) -> Sequence[Evidence]: ...


class RetrievalEnhancer:
    """Does not load models/indexes, diagnose failures, or decide sufficiency.

    Invalid caller inputs raise before any provider is called. Provider failures
    are recorded and safely contained; valid results from other calls survive.
    Calls that raise still spend budget, and no retry/recursive round is executed.
    """

    def __init__(self, retriever: RetrieverBackend, rewriter: QueryRewriter | None = None):
        self.retriever = retriever
        self.rewriter = rewriter if rewriter is not None else RuleQueryRewriter()
        cost = getattr(self.rewriter, "llm_calls_per_rewrite", None)
        if type(cost) is not int or cost != 0:
            raise ValueError("MVP rewriter must declare llm_calls_per_rewrite=0")

    def enhance(
        self, query: str, dataset: str, base_evidence: Sequence[Evidence],
        budget: EnhancementBudget | None = None,
    ) -> EnhancementResult:
        start = perf_counter()
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a nonempty string")
        if not isinstance(dataset, str) or not dataset.strip():
            raise ValueError("dataset must be a nonempty string")
        if not isinstance(base_evidence, Sequence) or isinstance(base_evidence, (str, bytes)):
            raise TypeError("base_evidence must be a sequence of evidence dictionaries")
        if len(base_evidence) > 5:
            raise ValueError("base_evidence must contain at most 5 items")
        if budget is None:
            budget = EnhancementBudget()
        if not isinstance(budget, EnhancementBudget):
            raise TypeError("budget must be an EnhancementBudget")
        base = deepcopy(list(base_evidence))
        # Reuse stage-one validation, including dataset and conflicting ID checks.
        fused = rrf_fuse(base, dataset=dataset)
        base_ids = {item["passage_id"] for item in fused["candidates"]}
        trace: EnhancementTrace = {
            "original_query": query, "rewritten_queries": [],
            "original_evidence_ids": [item["passage_id"] for item in base],
            "candidate_ids": [], "new_candidate_ids": [], "selected_ids": [],
            "retrieval_queries": [], "retrieval_calls": 0, "llm_calls": 0,
            "latency": 0.0, "status": "", "errors": [],
        }

        def finish(status: str) -> EnhancementResult:
            new_ids = [item["passage_id"] for item in fused["candidates"]
                       if item["passage_id"] not in base_ids]
            # Fallback preserves the exact original dictionaries and their order.
            selected = fused["evidence"] if new_ids else base
            trace["candidate_ids"] = [item["passage_id"] for item in fused["candidates"]]
            trace["new_candidate_ids"] = new_ids
            trace["selected_ids"] = [item["passage_id"] for item in selected]
            trace["status"] = status
            trace["latency"] = perf_counter() - start
            return {"evidence": deepcopy(selected), "candidates": deepcopy(fused["candidates"]),
                    "new_evidence_count": len(new_ids), "trace": deepcopy(trace)}

        if budget.max_retrieval_calls == 0:
            return finish("budget_exhausted")
        try:
            # Only query/dataset are provided: no Base text, Gold or answer records.
            trace["rewritten_queries"] = filter_rewrites(query, self.rewriter.rewrite(query, dataset))
        except Exception as exc:
            trace["errors"].append({"stage": "rewriting", "type": type(exc).__name__, "message": str(exc)})
            return finish("rewrite_failed")
        if not trace["rewritten_queries"]:
            return finish("no_rewrite")

        additional = []
        successes = 0
        for rewritten in trace["rewritten_queries"][:budget.max_retrieval_calls]:
            trace["retrieval_calls"] += 1
            trace["retrieval_queries"].append(rewritten)
            try:
                hits = self.retriever.retrieve(rewritten, dataset=dataset, top_k=budget.top_k_per_query)
                if not isinstance(hits, Sequence) or isinstance(hits, (str, bytes)):
                    raise TypeError("retriever output must be a sequence of evidence dictionaries")
                if len(hits) > budget.top_k_per_query:
                    raise ValueError("retriever returned more than the requested top_k")
                # Validate the entire call against Base and earlier successful calls.
                # Empty placeholders keep additional:N provenance aligned with
                # retrieval_queries even when an earlier call failed.
                trial = rrf_fuse(base, [*additional, hits], dataset=dataset)
                additional.append(deepcopy(list(hits)))
                fused = trial
                successes += 1
            except Exception as exc:
                additional.append([])
                trace["errors"].append({"stage": "retrieval", "query": rewritten,
                                        "type": type(exc).__name__, "message": str(exc)})
        if trace["errors"]:
            return finish("partial" if successes else "retrieval_failed")
        return finish("completed" if fused["new_evidence_count"] else "no_new_evidence")
