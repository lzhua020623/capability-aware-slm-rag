"""Test-only mock Enhancement implementations (P63).

Not registered by default. Do not use as production defaults.
These mocks never call models, networks, or teammate algorithms — they only
prove the EnhancementRequest -> EnhancementResult contract.
"""

from __future__ import annotations

from src.compensation.schema import EnhancementRequest, EnhancementResult, EnhancementStatus
from src.controller.schema import ControllerAction


class MockRetrievalEnhancement:
    """Placeholder for retrieval_enhancement lane."""

    def enhance(self, request: EnhancementRequest) -> EnhancementResult:
        return EnhancementResult(
            action=ControllerAction.RETRIEVAL_ENHANCEMENT,
            final_answer=request.initial_answer,
            evidence=request.evidence,
            changed_answer=False,
            status=EnhancementStatus.SUCCESS,
            metadata={"mock": "retrieval_enhancement", "note": "contract_only"},
            latency_ms=0.0,
        )


class MockEvidenceRefinementEnhancement:
    """Placeholder for evidence_refinement lane."""

    def enhance(self, request: EnhancementRequest) -> EnhancementResult:
        return EnhancementResult(
            action=ControllerAction.EVIDENCE_REFINEMENT,
            final_answer=request.initial_answer,
            evidence=request.evidence,
            changed_answer=False,
            status=EnhancementStatus.SUCCESS,
            metadata={"mock": "evidence_refinement", "note": "contract_only"},
            latency_ms=0.0,
        )


class MockGenerationRepairEnhancement:
    """Placeholder for generation_repair lane."""

    def enhance(self, request: EnhancementRequest) -> EnhancementResult:
        return EnhancementResult(
            action=ControllerAction.GENERATION_REPAIR,
            final_answer=request.initial_answer,
            evidence=request.evidence,
            changed_answer=False,
            status=EnhancementStatus.SUCCESS,
            metadata={"mock": "generation_repair", "note": "contract_only"},
            latency_ms=0.0,
        )


class RaisingEnhancement:
    """Test helper that always raises."""

    def enhance(self, request: EnhancementRequest) -> EnhancementResult:
        raise RuntimeError("mock_enhancement_failure")
