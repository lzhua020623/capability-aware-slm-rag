"""Adapter that plugs the rerank refiner into the Enhancement contract.

`EvidenceRefiner` speaks the earlier two-argument handoff signature
(`refiner(DiagnosisInput, ControllerDecision) -> str`). The host pipeline now
dispatches through `EnhancementExecutor`, which calls
`enhance(EnhancementRequest) -> EnhancementResult`
(`docs/ENHANCEMENT_INTEGRATION_CONTRACT.md`).

This adapter only converts schemas and forwards the call. It contains no
reranking, no thresholds and no acceptance logic: all of that stays in
`src/compensation/evidence_refinement.py`, so the Enhancement path and a direct
call produce the same answer.

Registration:

    executor = EnhancementExecutor()
    executor.register(ControllerAction.EVIDENCE_REFINEMENT, EvidenceRefinementAdapter(refiner))

Exceptions are deliberately not swallowed here: `EnhancementExecutor` already
turns them into an `ERROR` result that keeps the initial answer, which is the
safer behaviour and keeps the failure visible in the host's metadata.
"""

from __future__ import annotations

import time

from src.compensation.evidence_refinement import EvidenceRefiner
from src.compensation.refinement_schema import RefinementResult, RefinementStatus
from src.compensation.schema import EnhancementRequest, EnhancementResult, EnhancementStatus
from src.diagnosis.schema import DatasetName, DiagnosisInput

DEFAULT_DATASET = DatasetName.NQ
UNKNOWN_SAMPLE_ID = "unknown"


def request_to_diagnosis_input(request: EnhancementRequest) -> DiagnosisInput:
    """Rebuild the refiner's input from the request. Runtime fields only.

    `sample_id` is logging-only in both schemas, so a request without one gets a
    placeholder rather than failing. `dataset` falls back to NQ: the Controller
    routes no FEVER sample to this lane (FEVER has no accessibility samples),
    and the field only selects which frozen prompt is rebuilt.
    """
    if not isinstance(request, EnhancementRequest):
        raise TypeError(f"request must be an EnhancementRequest, got {type(request).__name__}")
    dataset = DatasetName(request.dataset) if request.dataset else DEFAULT_DATASET
    return DiagnosisInput(
        sample_id=request.sample_id or UNKNOWN_SAMPLE_ID,
        dataset=dataset,
        query=request.query,
        evidence=list(request.evidence),
        initial_answer=request.initial_answer,
    )


def refinement_metadata(result: RefinementResult) -> dict:
    """Runtime-visible trace of what the refiner did. No gold or offline fields."""
    return {
        "adapter": "evidence_refinement",
        "refinement_status": result.status.value,
        "evidence_changed": result.evidence_changed,
        "regenerated": result.regenerated,
        "reject_reason": result.reject_reason,
        "candidate_answer": result.candidate_answer,
        "used_probe_hint": result.used_probe_hint,
        "candidate_pool_size": result.candidate_pool_size,
        "dropped_passage_ids": list(result.dropped_passage_ids),
        "rerank_scores": list(result.rerank_scores),
    }


class EvidenceRefinementAdapter:
    """`Enhancement` lane for `ControllerAction.EVIDENCE_REFINEMENT`."""

    def __init__(self, refiner: EvidenceRefiner):
        if not isinstance(refiner, EvidenceRefiner):
            raise TypeError(f"refiner must be an EvidenceRefiner, got {type(refiner).__name__}")
        self.refiner = refiner

    def enhance(self, request: EnhancementRequest) -> EnhancementResult:
        item = request_to_diagnosis_input(request)
        start = time.perf_counter()
        result = self.refiner.run(item, request.controller_decision)
        latency_ms = (time.perf_counter() - start) * 1000.0
        # NOT_ROUTED can only happen if something dispatched a different action here;
        # report it as no_action rather than claiming the lane ran.
        status = (
            EnhancementStatus.NO_ACTION
            if result.status is RefinementStatus.NOT_ROUTED
            else EnhancementStatus.SUCCESS
        )
        return EnhancementResult(
            action=request.controller_decision.action,
            final_answer=result.final_answer,
            evidence=result.evidence,
            changed_answer=result.answer_changed,
            status=status,
            metadata=refinement_metadata(result),
            latency_ms=latency_ms,
        )
