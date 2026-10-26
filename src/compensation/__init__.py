"""Compensation / enhancement components.

Public contracts and executor live here. Mock implementations are under
``src.compensation.mocks`` and are **not** registered by default.
"""

from src.compensation.executor import ENHANCEMENT_ACTIONS, EnhancementExecutor
from src.compensation.interface import (
    Enhancement,
    EvidenceRefinementEnhancement,
    GenerationRepairEnhancement,
    RetrievalEnhancement,
)
from src.compensation.schema import EnhancementRequest, EnhancementResult, EnhancementStatus

__all__ = [
    "ENHANCEMENT_ACTIONS",
    "Enhancement",
    "EnhancementExecutor",
    "EnhancementRequest",
    "EnhancementResult",
    "EnhancementStatus",
    "EvidenceRefinementEnhancement",
    "GenerationRepairEnhancement",
    "RetrievalEnhancement",
]
