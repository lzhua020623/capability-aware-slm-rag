"""Independent retrieval enhancement; no model or FAISS dependencies."""

from .adapters import NQFaissBackend
from .enhancer import RetrievalEnhancer, RetrieverBackend
from .fusion import deduplicate_evidence, rrf_fuse
from .p3 import P320RetrievalEnhancer, PinnedScorer, build_c1_prompt, select_p3_20
from .rewriting import QueryRewriter, RuleQueryRewriter, filter_rewrites
from .types import EnhancementBudget, EnhancementResult, EnhancementTrace, Evidence, FusionResult

__all__ = [
    "Evidence", "FusionResult", "EnhancementBudget", "EnhancementResult", "EnhancementTrace",
    "QueryRewriter", "RuleQueryRewriter", "filter_rewrites", "RetrieverBackend", "RetrievalEnhancer",
    "deduplicate_evidence", "rrf_fuse", "NQFaissBackend",
    "P320RetrievalEnhancer", "PinnedScorer", "select_p3_20", "build_c1_prompt",
]
