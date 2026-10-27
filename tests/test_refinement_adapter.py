"""Tests for the Enhancement adapter around the rerank refiner. No model is loaded.

These cover the contract boundary: the adapter satisfies the Enhancement
protocol, converts schemas faithfully, and behaves correctly when dispatched
through the real EnhancementExecutor.
"""

from __future__ import annotations

import unittest

from src.compensation.evidence_refinement import EvidenceRefiner
from src.compensation.executor import EnhancementExecutor
from src.compensation.interface import Enhancement
from src.compensation.refinement_adapter import (
    EvidenceRefinementAdapter,
    refinement_metadata,
    request_to_diagnosis_input,
)
from src.compensation.refinement_schema import RefinementConfig
from src.compensation.schema import EnhancementRequest, EnhancementResult, EnhancementStatus
from src.controller.schema import ControllerAction, ControllerDecision
from src.diagnosis.labels import DiagnosisLabel
from src.diagnosis.schema import FORBIDDEN_RUNTIME_FIELDS, DatasetName, EvidenceItem

KEEP_ALL = RefinementConfig(keep_k=5)
PASSAGES = {
    "p1": "Christopher Marlowe was an English playwright who died in 1593 .",
    "p2": "Hamlet is a tragedy written by William Shakespeare around 1600 .",
}


def evidence() -> tuple[EvidenceItem, ...]:
    return tuple(
        EvidenceItem(rank=rank, passage_id=passage_id, text=PASSAGES[passage_id], score=1.0 - 0.1 * rank)
        for rank, passage_id in enumerate(PASSAGES, start=1)
    )


def decision(action: ControllerAction = ControllerAction.EVIDENCE_REFINEMENT) -> ControllerDecision:
    label = (
        DiagnosisLabel.EVIDENCE_ACCESSIBILITY
        if action is ControllerAction.EVIDENCE_REFINEMENT
        else DiagnosisLabel.CORRECT
    )
    return ControllerDecision(
        action=action,
        diagnosis_label=label,
        diagnosis_confidence=0.8,
        used_probe=False,
        probe_overrode_diagnosis=False,
        sample_id="s1",
    )


def request(**overrides) -> EnhancementRequest:
    fields = {
        "query": "who wrote hamlet",
        "initial_answer": "Christopher Marlowe",
        "evidence": evidence(),
        "controller_decision": decision(),
        "sample_id": "s1",
        "dataset": "nq",
    }
    fields.update(overrides)
    return EnhancementRequest.from_runtime(**fields)


class Fake:
    """Scorer + generator in one, scripted by passage id."""

    def __init__(self, scores: dict[str, float], answer: str = "William Shakespeare"):
        self._scores = {PASSAGES[key]: value for key, value in scores.items()}
        self._answer = answer
        self.calls = 0

    def score(self, query, texts):
        return [self._scores[text] for text in texts]

    def generate(self, prompt):
        self.calls += 1
        return self._answer

    def prompt_builder(self, item, passages):
        return "|".join(passage.passage_id for passage in passages)


def adapter_for(fake: Fake, config: RefinementConfig = KEEP_ALL) -> EvidenceRefinementAdapter:
    refiner = EvidenceRefiner(fake.score, fake.generate, config=config, prompt_builder=fake.prompt_builder)
    return EvidenceRefinementAdapter(refiner)


class ProtocolTest(unittest.TestCase):
    def test_adapter_satisfies_the_enhancement_protocol(self):
        self.assertIsInstance(adapter_for(Fake({"p1": 0.1, "p2": 0.9})), Enhancement)

    def test_adapter_requires_a_refiner(self):
        with self.assertRaises(TypeError):
            EvidenceRefinementAdapter("not a refiner")


class RequestConversionTest(unittest.TestCase):
    def test_runtime_fields_are_carried_over(self):
        item = request_to_diagnosis_input(request())
        self.assertEqual(item.sample_id, "s1")
        self.assertEqual(item.dataset, DatasetName.NQ)
        self.assertEqual(item.query, "who wrote hamlet")
        self.assertEqual(item.initial_answer, "Christopher Marlowe")
        self.assertEqual([passage.passage_id for passage in item.evidence], ["p1", "p2"])

    def test_sample_id_falls_back_to_the_controller_decision(self):
        # EnhancementRequest.from_runtime already borrows the decision's sample_id.
        self.assertEqual(request_to_diagnosis_input(request(sample_id=None)).sample_id, "s1")

    def test_missing_sample_id_and_dataset_fall_back(self):
        anonymous = decision()
        object.__setattr__(anonymous, "sample_id", None)
        item = request_to_diagnosis_input(request(sample_id=None, dataset=None, controller_decision=anonymous))
        self.assertEqual(item.sample_id, "unknown")
        self.assertEqual(item.dataset, DatasetName.NQ)

    def test_fever_dataset_is_honoured(self):
        self.assertEqual(request_to_diagnosis_input(request(dataset="fever")).dataset, DatasetName.FEVER)

    def test_a_non_request_is_rejected(self):
        with self.assertRaises(TypeError):
            request_to_diagnosis_input({"query": "who wrote hamlet"})


class EnhanceTest(unittest.TestCase):
    def test_replaced_answer_is_reported_as_a_change(self):
        fake = Fake({"p1": 0.1, "p2": 0.9})
        result = adapter_for(fake).enhance(request())
        self.assertIsInstance(result, EnhancementResult)
        self.assertEqual(result.action, ControllerAction.EVIDENCE_REFINEMENT)
        self.assertEqual(result.final_answer, "William Shakespeare")
        self.assertTrue(result.changed_answer)
        self.assertEqual(result.status, EnhancementStatus.SUCCESS)
        self.assertEqual([passage.passage_id for passage in result.evidence], ["p2", "p1"])
        self.assertEqual(result.metadata["refinement_status"], "answer_replaced")
        self.assertGreaterEqual(result.latency_ms, 0.0)

    def test_kept_original_is_still_a_successful_run(self):
        fake = Fake({"p1": 0.1, "p2": 0.9}, answer="Ben Jonson")
        result = adapter_for(fake).enhance(request())
        self.assertEqual(result.final_answer, "Christopher Marlowe")
        self.assertFalse(result.changed_answer)
        self.assertEqual(result.status, EnhancementStatus.SUCCESS)
        self.assertEqual(result.metadata["refinement_status"], "kept_original")
        self.assertEqual(result.metadata["reject_reason"], "not_grounded_in_kept_evidence")
        self.assertEqual(result.metadata["candidate_answer"], "Ben Jonson")

    def test_unchanged_order_spends_no_generation_call(self):
        fake = Fake({"p1": 0.9, "p2": 0.1})
        result = adapter_for(fake).enhance(request())
        self.assertEqual(result.metadata["refinement_status"], "no_change")
        self.assertFalse(result.changed_answer)
        self.assertEqual(fake.calls, 0)

    def test_a_foreign_action_is_reported_as_no_action(self):
        fake = Fake({"p1": 0.1, "p2": 0.9})
        result = adapter_for(fake).enhance(request(controller_decision=decision(ControllerAction.NO_ACTION)))
        self.assertEqual(result.status, EnhancementStatus.NO_ACTION)
        self.assertEqual(result.final_answer, "Christopher Marlowe")
        self.assertEqual(fake.calls, 0)

    def test_metadata_carries_no_offline_fields(self):
        fake = Fake({"p1": 0.1, "p2": 0.9})
        metadata = adapter_for(fake).enhance(request()).metadata
        self.assertEqual(set(metadata) & FORBIDDEN_RUNTIME_FIELDS, set())

    def test_metadata_helper_shape(self):
        fake = Fake({"p1": 0.1, "p2": 0.9})
        refiner = EvidenceRefiner(fake.score, fake.generate, config=KEEP_ALL, prompt_builder=fake.prompt_builder)
        metadata = refinement_metadata(refiner.run(request_to_diagnosis_input(request()), decision()))
        self.assertEqual(metadata["adapter"], "evidence_refinement")
        self.assertIn("rerank_scores", metadata)


class ExecutorIntegrationTest(unittest.TestCase):
    def test_executor_dispatches_to_the_registered_adapter(self):
        fake = Fake({"p1": 0.1, "p2": 0.9})
        executor = EnhancementExecutor()
        executor.register(ControllerAction.EVIDENCE_REFINEMENT, adapter_for(fake))
        result = executor.execute(request())
        self.assertEqual(result.final_answer, "William Shakespeare")
        self.assertEqual(result.status, EnhancementStatus.SUCCESS)
        self.assertIn(ControllerAction.EVIDENCE_REFINEMENT, executor.registered_actions())

    def test_a_crashing_refiner_keeps_the_initial_answer(self):
        class Exploding:
            def enhance(self, request):
                raise RuntimeError("boom")

        executor = EnhancementExecutor()
        executor.register(ControllerAction.EVIDENCE_REFINEMENT, Exploding())
        result = executor.execute(request())
        self.assertEqual(result.status, EnhancementStatus.ERROR)
        self.assertEqual(result.final_answer, "Christopher Marlowe")
        self.assertFalse(result.changed_answer)

    def test_an_unregistered_lane_is_unavailable(self):
        result = EnhancementExecutor().execute(request())
        self.assertEqual(result.status, EnhancementStatus.UNAVAILABLE)
        self.assertEqual(result.final_answer, "Christopher Marlowe")

    def test_the_adapter_and_a_direct_call_agree(self):
        fake_direct = Fake({"p1": 0.1, "p2": 0.9})
        refiner = EvidenceRefiner(
            fake_direct.score, fake_direct.generate, config=KEEP_ALL, prompt_builder=fake_direct.prompt_builder
        )
        direct = refiner(request_to_diagnosis_input(request()), decision())
        through_executor = EvidenceRefinementAdapter(refiner).enhance(request()).final_answer
        self.assertEqual(direct, through_executor)


if __name__ == "__main__":
    unittest.main()
