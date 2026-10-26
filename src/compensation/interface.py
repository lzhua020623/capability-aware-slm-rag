"""Virtual Enhancement interfaces (P63).

Controller decides *what* to do. Implementations execute algorithms.
Future teammate modules must be wrapped by thin adapters that only convert
schemas / forward resources — adapters must not rewrite teammate algorithms.

  TeamRetrievalModule  -> RetrievalEnhancementAdapter -> Enhancement.enhance(...)
  TeamEvidenceModule   -> EvidenceRefinementAdapter   -> Enhancement.enhance(...)
  TeamGenerationModule -> GenerationRepairAdapter     -> Enhancement.enhance(...)
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.compensation.schema import EnhancementRequest, EnhancementResult


@runtime_checkable
class Enhancement(Protocol):
    """Shared protocol for all three Enhancement lanes."""

    def enhance(self, request: EnhancementRequest) -> EnhancementResult:
        """Run one Enhancement call. Must not read gold / offline labels."""
        ...


# Logical role markers (same Protocol; distinguished by registered ControllerAction).
class RetrievalEnhancement(Enhancement, Protocol):
    """Lane for ControllerAction.RETRIEVAL_ENHANCEMENT (evidence_insufficiency)."""


class EvidenceRefinementEnhancement(Enhancement, Protocol):
    """Lane for ControllerAction.EVIDENCE_REFINEMENT (evidence_accessibility)."""


class GenerationRepairEnhancement(Enhancement, Protocol):
    """Lane for ControllerAction.GENERATION_REPAIR (evidence_utilisation)."""
