"""End-to-end Adaptive RAG host pipeline (P64).

Wiring only — reuses Diagnosis, FrozenOnlineSelectionAdapter, Controller.decide,
and EnhancementExecutor. Probe stays OFF on this path. Concrete Enhancement
implementations are injected via the executor registry (mocks or future adapters).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any, Protocol

import numpy as np

from src.compensation.executor import EnhancementExecutor
from src.compensation.schema import EnhancementRequest, EnhancementStatus
from src.controller.controller import decide
from src.controller.schema import SelectionDecision
from src.controller.selection import FrozenOnlineSelectionAdapter
from src.diagnosis.hierarchical_classifier import HierarchicalDiagnosis, HierarchicalDiagnosisClassifier
from src.diagnosis.schema import DiagnosisInput
from src.pipeline.schema import AdaptiveRAGResult, PipelineStatus


class ReferenceRAG(Protocol):
    """Produces Reference RAG C1-equivalent DiagnosisInput for one query."""

    def run(
        self,
        query: str,
        *,
        sample_id: str | None = None,
        dataset: str = "nq",
    ) -> DiagnosisInput: ...


class FeatureProvider(Protocol):
    """Builds the frozen adapter feature row for one item (lifecycle owned by host)."""

    def features_for(
        self,
        item: DiagnosisInput,
        diagnosis: HierarchicalDiagnosis,
    ) -> np.ndarray | Sequence[float]: ...


class SelectionProvider(Protocol):
    """Optional override for tests; production uses FrozenOnlineSelectionAdapter."""

    def select(
        self,
        diagnosis: HierarchicalDiagnosis,
        adapter_features: np.ndarray | Sequence[float],
        *,
        sample_id: str | None = None,
    ) -> SelectionDecision: ...


def _pipeline_status(enhancement_status: EnhancementStatus) -> PipelineStatus:
    if enhancement_status is EnhancementStatus.NO_ACTION:
        return PipelineStatus.NO_ACTION
    if enhancement_status is EnhancementStatus.UNAVAILABLE:
        return PipelineStatus.ENHANCEMENT_UNAVAILABLE
    if enhancement_status is EnhancementStatus.ERROR:
        return PipelineStatus.ENHANCEMENT_ERROR
    return PipelineStatus.SUCCESS


class AdaptiveRAGPipeline:
    """Host orchestration with injected resources (no per-query model reload)."""

    def __init__(
        self,
        *,
        reference_rag: ReferenceRAG | None = None,
        classifier: HierarchicalDiagnosisClassifier | Any,
        selection_adapter: FrozenOnlineSelectionAdapter | SelectionProvider,
        feature_provider: FeatureProvider,
        executor: EnhancementExecutor,
    ) -> None:
        self.reference_rag = reference_rag
        self.classifier = classifier
        self.selection_adapter = selection_adapter
        self.feature_provider = feature_provider
        self.executor = executor

    def run(
        self,
        query: str,
        *,
        sample_id: str | None = None,
        dataset: str = "nq",
    ) -> AdaptiveRAGResult:
        if self.reference_rag is None:
            raise ValueError("reference_rag is required for AdaptiveRAGPipeline.run(query)")
        item = self.reference_rag.run(query, sample_id=sample_id, dataset=dataset)
        return self.run_from_item(item)

    def run_from_item(self, item: DiagnosisInput) -> AdaptiveRAGResult:
        """Continue from an existing Reference RAG / C1 DiagnosisInput (Probe OFF)."""
        if not isinstance(item, DiagnosisInput):
            raise TypeError(f"item must be DiagnosisInput, got {type(item).__name__}")
        started = time.perf_counter()
        diagnosis = self.classifier.diagnose(item)
        features = self.feature_provider.features_for(item, diagnosis)
        return self.run_from_prepared(item, diagnosis, features, started=started)

    def run_from_prepared(
        self,
        item: DiagnosisInput,
        diagnosis: HierarchicalDiagnosis,
        features: np.ndarray | Sequence[float],
        *,
        started: float | None = None,
    ) -> AdaptiveRAGResult:
        """Select → Controller → Enhancement using precomputed diagnosis/features."""
        if not isinstance(item, DiagnosisInput):
            raise TypeError(f"item must be DiagnosisInput, got {type(item).__name__}")
        if not isinstance(diagnosis, HierarchicalDiagnosis):
            raise TypeError(
                f"diagnosis must be HierarchicalDiagnosis, got {type(diagnosis).__name__}"
            )
        t0 = time.perf_counter() if started is None else started
        selection = self.selection_adapter.select(
            diagnosis, features, sample_id=item.sample_id
        )
        if not isinstance(selection, SelectionDecision):
            raise TypeError("selection_adapter.select must return SelectionDecision")
        if selection.used_probe:
            raise ValueError("adaptive pipeline requires used_probe=False")

        controller_decision = decide(selection)
        if controller_decision.used_probe:
            raise ValueError("adaptive pipeline requires ControllerDecision.used_probe=False")

        request = EnhancementRequest.from_runtime(
            query=item.query,
            initial_answer=item.initial_answer,
            evidence=item.evidence,
            controller_decision=controller_decision,
            sample_id=item.sample_id,
            dataset=item.dataset.value,
        )
        enhancement = self.executor.execute(request)

        status = _pipeline_status(enhancement.status)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return AdaptiveRAGResult(
            query=item.query,
            initial_answer=item.initial_answer,
            final_answer=enhancement.final_answer,
            initial_evidence=tuple(item.evidence),
            final_evidence=enhancement.evidence,
            diagnosis=diagnosis,
            selection_decision=selection,
            controller_decision=controller_decision,
            enhancement_result=enhancement,
            changed_answer=enhancement.changed_answer,
            status=status,
            sample_id=item.sample_id,
            metadata={
                "probe": "OFF",
                "used_probe": False,
                "controller_action": controller_decision.action.value,
                "enhancement_status": enhancement.status.value,
            },
            latency_ms=latency_ms,
        )


def run_adaptive_rag(
    query: str,
    *,
    pipeline: AdaptiveRAGPipeline,
    sample_id: str | None = None,
    dataset: str = "nq",
) -> AdaptiveRAGResult:
    """Public functional entry for the adaptive host pipeline."""
    return pipeline.run(query, sample_id=sample_id, dataset=dataset)


def run_adaptive_rag_from_item(
    item: DiagnosisInput,
    *,
    pipeline: AdaptiveRAGPipeline,
) -> AdaptiveRAGResult:
    """Public entry when Reference RAG output is already a DiagnosisInput."""
    return pipeline.run_from_item(item)


def build_adaptive_pipeline(
    *,
    classifier: HierarchicalDiagnosisClassifier | Any,
    selection_adapter: FrozenOnlineSelectionAdapter | SelectionProvider,
    feature_provider: FeatureProvider,
    executor: EnhancementExecutor | None = None,
    reference_rag: ReferenceRAG | None = None,
) -> AdaptiveRAGPipeline:
    """Factory: empty executor by default (missing Enhancement -> safe unavailable)."""
    return AdaptiveRAGPipeline(
        reference_rag=reference_rag,
        classifier=classifier,
        selection_adapter=selection_adapter,
        feature_provider=feature_provider,
        executor=EnhancementExecutor() if executor is None else executor,
    )
