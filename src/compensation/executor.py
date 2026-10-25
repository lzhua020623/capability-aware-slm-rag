"""EnhancementExecutor: dispatch ControllerDecision.action to registered modules.

Does not re-run Diagnosis, selection, or subtype routing. Missing or failing
implementations fall back to the initial answer (safe).
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping

from src.compensation.interface import Enhancement
from src.compensation.schema import EnhancementRequest, EnhancementResult, EnhancementStatus
from src.controller.schema import ControllerAction


ENHANCEMENT_ACTIONS: frozenset[ControllerAction] = frozenset(
    {
        ControllerAction.RETRIEVAL_ENHANCEMENT,
        ControllerAction.EVIDENCE_REFINEMENT,
        ControllerAction.GENERATION_REPAIR,
    }
)


def _keep_initial(
    request: EnhancementRequest,
    *,
    status: EnhancementStatus,
    metadata: Mapping[str, object],
    error: str | None = None,
) -> EnhancementResult:
    return EnhancementResult(
        action=request.controller_decision.action,
        final_answer=request.initial_answer,
        evidence=request.evidence,
        changed_answer=False,
        status=status,
        metadata=dict(metadata),
        latency_ms=0.0,
        error=error,
    )


class EnhancementExecutor:
    """Registry + dispatcher. No teammate class names are hard-coded."""

    def __init__(self) -> None:
        self._registry: MutableMapping[ControllerAction, Enhancement] = {}

    def register(self, action: ControllerAction | str, implementation: Enhancement) -> None:
        action = ControllerAction(action)
        if action is ControllerAction.NO_ACTION:
            raise ValueError("no_action has no Enhancement implementation; handled by the executor")
        if action not in ENHANCEMENT_ACTIONS:
            raise ValueError(f"Unsupported enhancement action: {action}")
        if not callable(getattr(implementation, "enhance", None)):
            raise TypeError("implementation must provide enhance(request) -> EnhancementResult")
        self._registry[action] = implementation

    def unregister(self, action: ControllerAction | str) -> None:
        action = ControllerAction(action)
        self._registry.pop(action, None)

    def registered_actions(self) -> frozenset[ControllerAction]:
        return frozenset(self._registry)

    def execute(self, request: EnhancementRequest) -> EnhancementResult:
        """Dispatch by ControllerDecision.action only — never re-diagnose."""
        if not isinstance(request, EnhancementRequest):
            raise TypeError(f"request must be EnhancementRequest, got {type(request).__name__}")
        action = request.controller_decision.action

        if action is ControllerAction.NO_ACTION:
            return _keep_initial(
                request,
                status=EnhancementStatus.NO_ACTION,
                metadata={"reason": "controller_no_action"},
            )

        if action not in ENHANCEMENT_ACTIONS:
            return _keep_initial(
                request,
                status=EnhancementStatus.UNAVAILABLE,
                metadata={"enhancement_not_registered": True, "reason": f"unsupported_action:{action}"},
                error=f"unsupported_action:{action}",
            )

        implementation = self._registry.get(action)
        if implementation is None:
            return _keep_initial(
                request,
                status=EnhancementStatus.UNAVAILABLE,
                metadata={
                    "enhancement_not_registered": True,
                    "action": action.value,
                },
                error="enhancement_not_registered",
            )

        try:
            result = implementation.enhance(request)
        except Exception as exc:  # noqa: BLE001 — host must not crash on module failure
            return _keep_initial(
                request,
                status=EnhancementStatus.ERROR,
                metadata={
                    "enhancement_exception": True,
                    "action": action.value,
                    "exception_type": type(exc).__name__,
                },
                error=f"{type(exc).__name__}: {exc}",
            )

        if not isinstance(result, EnhancementResult):
            return _keep_initial(
                request,
                status=EnhancementStatus.ERROR,
                metadata={"enhancement_invalid_result": True, "action": action.value},
                error="implementation returned non-EnhancementResult",
            )
        mismatch = None
        if result.action is not action:
            mismatch = "result_action_mismatch"
        elif result.changed_answer != (result.final_answer != request.initial_answer):
            mismatch = "result_changed_answer_mismatch"
        elif result.status is not EnhancementStatus.SUCCESS and (
                result.final_answer != request.initial_answer or result.evidence != request.evidence):
            mismatch = "non_success_changed_entering_state"
        if mismatch:
            return _keep_initial(
                request, status=EnhancementStatus.ERROR,
                metadata={"enhancement_invalid_result": True, "reason": mismatch,
                          "action": action.value}, error=mismatch)
        return result
