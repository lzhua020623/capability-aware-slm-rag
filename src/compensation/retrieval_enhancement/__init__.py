"""Independent retrieval enhancement; no model or FAISS dependencies."""

from .enhancer import RetrievalEnhancer, RetrieverBackend
from .fusion import deduplicate_evidence, rrf_fuse
from .rewriting import QueryRewriter, RuleQueryRewriter, filter_rewrites
from .types import EnhancementBudget, EnhancementResult, EnhancementTrace, Evidence, FusionResult

__all__ = [
    "Evidence", "FusionResult", "EnhancementBudget", "EnhancementResult", "EnhancementTrace",
    "QueryRewriter", "RuleQueryRewriter", "filter_rewrites", "RetrieverBackend", "RetrievalEnhancer",
    "deduplicate_evidence", "rrf_fuse",
]
