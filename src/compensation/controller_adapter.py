"""E's boundary with the frozen ControllerDecision; no routing policy here.

The upstream Controller owns Diagnosis, Probe and Safe Binary resolution.
This module accepts its resolved decision and executes only generation_repair.
"""

from __future__ import annotations

import math
from typing import Protocol

from src.compensation.answer_repair import (
    DEFAULT_MAX_NEW_TOKENS, FEVER_ONLY_STRATEGIES, LINE_OUTPUT_STRATEGIES, NLI_STRATEGIES, NQ_ONLY_STRATEGIES,
    RepairGenerator, RepairResult, repair_answer, V8_STRATEGIES,
)
from src.diagnosis import DatasetName, DiagnosisInput


RUNTIME_STRATEGIES = ("verify_repair", "grounded_reanswer", "query_grounded", "guarded_reanswer",
                      "extractive_reanswer", "fever_nli", "short_answer_quote", "fever_nli_anchored", *V8_STRATEGIES)


class ResolvedDecision(Protocol):
    action: str
    diagnosis_label: str
    diagnosis_confidence: float
    used_probe: bool
    probe_overrode_diagnosis: bool
    needs_compensation: bool
    sample_id: str | None


def controller_route_status(request: DiagnosisInput, decision: ResolvedDecision | None) -> str | None:
    if decision is None:
        raise ValueError("Missing resolved ControllerDecision")
    action = str(decision.action)
    if action not in ("no_action", "retrieval_enhancement", "evidence_refinement", "generation_repair"):
        raise ValueError("Unknown Controller action")
    if decision.sample_id is not None and str(decision.sample_id) != request.sample_id:
        raise ValueError("Controller decision and input sample IDs differ")
    confidence = decision.diagnosis_confidence
    if (isinstance(confidence, bool) or not isinstance(confidence, (int, float))
            or not math.isfinite(confidence) or not 0 <= confidence <= 1):
        raise ValueError("Invalid Controller confidence")
    for name in ("used_probe", "probe_overrode_diagnosis", "needs_compensation"):
        if not isinstance(getattr(decision, name), bool):
            raise ValueError(f"Controller {name} must be bool")
    if decision.probe_overrode_diagnosis and not decision.used_probe:
        raise ValueError("Controller override requires a completed Probe")
    if decision.needs_compensation != (action != "no_action"):
        raise ValueError("Controller needs_compensation disagrees with action")
    if action != "generation_repair":
        return "skipped_controller_action"
    if str(decision.diagnosis_label) != "evidence_utilisation":
        raise ValueError("generation_repair requires the resolved utilisation label")
    # No confidence threshold, needs_probe gate, or second Diagnosis decision:
    # the Controller has already completed those steps.
    return None


def generation_repair(
    request: DiagnosisInput,
    decision: ResolvedDecision,
    generator: RepairGenerator | None = None,
    *,
    strategy: str = "verify_repair",
    verifier=None,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> RepairResult:
    status = controller_route_status(request, decision)
    if status is not None:
        return RepairResult(request.initial_answer, request.initial_answer, status)
    if strategy not in RUNTIME_STRATEGIES:
        raise ValueError("Unknown E runtime strategy")
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be a positive int")
    if (strategy in FEVER_ONLY_STRATEGIES and request.dataset != DatasetName.FEVER or
            strategy in NQ_ONLY_STRATEGIES and request.dataset != DatasetName.NQ):
        raise ValueError("The selected strategy does not support this dataset")
    if not any(item.text.strip() for item in request.evidence):
        return RepairResult(request.initial_answer, request.initial_answer, "skipped_no_evidence")
    if strategy in NLI_STRATEGIES:
        if verifier is None:
            from src.compensation.nli_repair import MNLIVerifier
            verifier = MNLIVerifier()
        return repair_answer(request, strategy=strategy, verifier=verifier)
    if generator is None:
        from src.compensation.qwen_repair import QwenRepairGenerator
        generator = QwenRepairGenerator(json_output=strategy not in LINE_OUTPUT_STRATEGIES)
    return repair_answer(request, generator, strategy=strategy, max_new_tokens=max_new_tokens)
