"""P63 virtual Enhancement interface + executor tests. No models, no Test data, no gold."""

from __future__ import annotations

import importlib
import inspect
import unittest

from src.compensation.executor import EnhancementExecutor
from src.compensation.mocks import (
    MockEvidenceRefinementEnhancement,
    MockGenerationRepairEnhancement,
    MockRetrievalEnhancement,
    RaisingEnhancement,
)
from src.compensation.schema import EnhancementRequest, EnhancementResult, EnhancementStatus
from src.controller.controller import decide
from src.controller.schema import ControllerAction, ControllerDecision, SelectionDecision, SelectionLane
from src.diagnosis.labels import DiagnosisLabel
from src.diagnosis.schema import EvidenceItem
from tests.test_probe_mapping import nq_input


def evidence_tuple():
    return (
        EvidenceItem(1, "p1", "alpha text", 0.9),
        EvidenceItem(2, "p2", "beta text", 0.8),
    )


def decision_for(action: ControllerAction, label: DiagnosisLabel) -> ControllerDecision:
    if action is ControllerAction.NO_ACTION:
        sel = SelectionDecision(
            False, label, 0.1, SelectionLane.NO_ACTION, 0.5, 0.4, 0.1, False, "fix"
        )
    else:
        lane = (
            SelectionLane.INSUFF_PRESERVE
            if label is DiagnosisLabel.EVIDENCE_INSUFFICIENCY
            else SelectionLane.RESIDUAL_COMBO
        )
        sel = SelectionDecision(True, label, 0.8, lane, 0.5, 0.8, 0.6, False, "fix")
    return decide(sel)


def request_for(action: ControllerAction, label: DiagnosisLabel) -> EnhancementRequest:
    decision = decision_for(action, label)
    # Force action match for no_action / routing consistency when label differs
    if action is ControllerAction.NO_ACTION:
        decision = ControllerDecision(
            ControllerAction.NO_ACTION,
            label,
            0.5,
            False,
            False,
            "fix",
            selection_score=0.1,
            selection_lane=SelectionLane.NO_ACTION,
            reason="not_selected",
        )
    return EnhancementRequest.from_runtime(
        query="what is x",
        initial_answer="original",
        evidence=evidence_tuple(),
        controller_decision=decision,
        sample_id="fx1",
        dataset="nq",
    )


class ContractSchemaTests(unittest.TestCase):
    def test_request_rejects_gold_in_context(self):
        decision = decision_for(
            ControllerAction.RETRIEVAL_ENHANCEMENT, DiagnosisLabel.EVIDENCE_INSUFFICIENCY
        )
        with self.assertRaises(ValueError):
            EnhancementRequest.from_runtime(
                query="q",
                initial_answer="a",
                evidence=evidence_tuple(),
                controller_decision=decision,
                context={"gold_answer": "secret"},
            )

    def test_no_gold_in_public_signatures(self):
        for fn in (EnhancementRequest.from_runtime, EnhancementExecutor.execute):
            names = set(inspect.signature(fn).parameters)
            self.assertFalse(names & {"gold_answer", "gold_evidence", "target_label"})


class ExecutorBehaviorTests(unittest.TestCase):
    def setUp(self):
        self.executor = EnhancementExecutor()
        self.retrieval = MockRetrievalEnhancement()
        self.evidence = MockEvidenceRefinementEnhancement()
        self.generation = MockGenerationRepairEnhancement()

    def test_no_action(self):
        req = request_for(ControllerAction.NO_ACTION, DiagnosisLabel.EVIDENCE_UTILISATION)
        result = self.executor.execute(req)
        self.assertIs(result.status, EnhancementStatus.NO_ACTION)
        self.assertEqual(result.final_answer, "original")
        self.assertEqual(result.evidence, evidence_tuple())
        self.assertFalse(result.changed_answer)
        self.assertEqual(self.executor.registered_actions(), frozenset())

    def test_retrieval_mock(self):
        self.executor.register(ControllerAction.RETRIEVAL_ENHANCEMENT, self.retrieval)
        req = request_for(ControllerAction.RETRIEVAL_ENHANCEMENT, DiagnosisLabel.EVIDENCE_INSUFFICIENCY)
        result = self.executor.execute(req)
        self.assertIs(result.status, EnhancementStatus.SUCCESS)
        self.assertIs(result.action, ControllerAction.RETRIEVAL_ENHANCEMENT)
        self.assertEqual(result.metadata.get("mock"), "retrieval_enhancement")
        self.assertEqual(result.final_answer, req.initial_answer)

    def test_evidence_refinement_mock(self):
        self.executor.register(ControllerAction.EVIDENCE_REFINEMENT, self.evidence)
        req = request_for(ControllerAction.EVIDENCE_REFINEMENT, DiagnosisLabel.EVIDENCE_ACCESSIBILITY)
        result = self.executor.execute(req)
        self.assertEqual(result.metadata.get("mock"), "evidence_refinement")
        self.assertIs(result.action, ControllerAction.EVIDENCE_REFINEMENT)

    def test_generation_repair_mock(self):
        self.executor.register(ControllerAction.GENERATION_REPAIR, self.generation)
        req = request_for(ControllerAction.GENERATION_REPAIR, DiagnosisLabel.EVIDENCE_UTILISATION)
        result = self.executor.execute(req)
        self.assertEqual(result.metadata.get("mock"), "generation_repair")
        self.assertIs(result.action, ControllerAction.GENERATION_REPAIR)

    def test_only_matching_implementation_called(self):
        calls = []

        class Tracking:
            def __init__(self, name):
                self.name = name

            def enhance(self, request):
                calls.append(self.name)
                return EnhancementResult(
                    action=request.controller_decision.action,
                    final_answer=request.initial_answer,
                    evidence=request.evidence,
                    changed_answer=False,
                    status=EnhancementStatus.SUCCESS,
                    metadata={"mock": self.name},
                )

        self.executor.register(ControllerAction.RETRIEVAL_ENHANCEMENT, Tracking("R"))
        self.executor.register(ControllerAction.EVIDENCE_REFINEMENT, Tracking("E"))
        self.executor.register(ControllerAction.GENERATION_REPAIR, Tracking("G"))
        self.executor.execute(
            request_for(ControllerAction.EVIDENCE_REFINEMENT, DiagnosisLabel.EVIDENCE_ACCESSIBILITY)
        )
        self.assertEqual(calls, ["E"])

    def test_request_fields_passed(self):
        seen = {}

        class Capture:
            def enhance(self, request):
                seen["query"] = request.query
                seen["initial_answer"] = request.initial_answer
                seen["n_evidence"] = len(request.evidence)
                seen["failure_type"] = request.failure_type
                seen["action"] = request.controller_decision.action
                return EnhancementResult(
                    action=request.controller_decision.action,
                    final_answer=request.initial_answer,
                    evidence=request.evidence,
                    changed_answer=False,
                    status=EnhancementStatus.SUCCESS,
                )

        self.executor.register(ControllerAction.RETRIEVAL_ENHANCEMENT, Capture())
        req = request_for(ControllerAction.RETRIEVAL_ENHANCEMENT, DiagnosisLabel.EVIDENCE_INSUFFICIENCY)
        self.executor.execute(req)
        self.assertEqual(seen["query"], "what is x")
        self.assertEqual(seen["initial_answer"], "original")
        self.assertEqual(seen["n_evidence"], 2)
        self.assertIs(seen["failure_type"], DiagnosisLabel.EVIDENCE_INSUFFICIENCY)
        self.assertIs(seen["action"], ControllerAction.RETRIEVAL_ENHANCEMENT)

    def test_result_standardized(self):
        self.executor.register(ControllerAction.GENERATION_REPAIR, self.generation)
        result = self.executor.execute(
            request_for(ControllerAction.GENERATION_REPAIR, DiagnosisLabel.EVIDENCE_UTILISATION)
        )
        self.assertIsInstance(result, EnhancementResult)
        payload = result.to_dict()
        for key in ("action", "final_answer", "evidence", "changed_answer", "status", "metadata"):
            self.assertIn(key, payload)

    def test_unregistered_safe_fallback(self):
        req = request_for(ControllerAction.RETRIEVAL_ENHANCEMENT, DiagnosisLabel.EVIDENCE_INSUFFICIENCY)
        result = self.executor.execute(req)
        self.assertIs(result.status, EnhancementStatus.UNAVAILABLE)
        self.assertFalse(result.changed_answer)
        self.assertEqual(result.final_answer, "original")
        self.assertTrue(result.metadata.get("enhancement_not_registered"))

    def test_exception_safe_fallback(self):
        self.executor.register(ControllerAction.RETRIEVAL_ENHANCEMENT, RaisingEnhancement())
        result = self.executor.execute(
            request_for(ControllerAction.RETRIEVAL_ENHANCEMENT, DiagnosisLabel.EVIDENCE_INSUFFICIENCY)
        )
        self.assertIs(result.status, EnhancementStatus.ERROR)
        self.assertFalse(result.changed_answer)
        self.assertEqual(result.final_answer, "original")
        self.assertIn("RuntimeError", result.error or "")

    def test_cannot_register_no_action(self):
        with self.assertRaises(ValueError):
            self.executor.register(ControllerAction.NO_ACTION, self.retrieval)

    def test_mocks_not_default_registered(self):
        self.assertEqual(EnhancementExecutor().registered_actions(), frozenset())

    def test_probe_irrelevant(self):
        # ControllerDecision on frozen path always used_probe=False; executor ignores Probe.
        req = request_for(ControllerAction.NO_ACTION, DiagnosisLabel.CORRECT)
        self.assertFalse(req.controller_decision.used_probe)
        result = self.executor.execute(req)
        self.assertIs(result.status, EnhancementStatus.NO_ACTION)


class IsolationTests(unittest.TestCase):
    def test_controller_does_not_import_compensation_implementations(self):
        from pathlib import Path

        path = Path(importlib.import_module("src.controller.controller").__file__)
        text = path.read_text(encoding="utf-8")
        self.assertNotIn("src.compensation", text)

    def test_nq_input_still_usable_without_enhancement(self):
        item = nq_input()
        self.assertTrue(item.initial_answer or item.initial_answer == "")


if __name__ == "__main__":
    unittest.main()
