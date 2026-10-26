"""Tests for the evidence_refinement action: rerank, re-ask once, accept only if grounded.

No model is loaded: the scorer, the generator and (mostly) the prompt builder
are injected. One test exercises the real frozen prompt builders.
"""

from __future__ import annotations

import unittest

from src.compensation.evidence_refinement import (
    EvidenceRefiner,
    build_refinement_prompt,
    evidence_refinement,
    is_grounded,
    is_refusal,
    rejection_reason,
)
from src.compensation.schema import RefinementConfig, RefinementResult, RefinementStatus
from src.controller.schema import ControllerAction, ControllerDecision
from src.diagnosis.labels import DiagnosisLabel
from src.diagnosis.schema import DatasetName, DiagnosisInput, EvidenceItem
from src.probe.schema import ProbeResult

KEEP_ALL = RefinementConfig(keep_k=5)
KEEP_TWO = RefinementConfig(keep_k=2)

PASSAGES = {
    "p1": "Christopher Marlowe was an English playwright who died in 1593 .",
    "p2": "Hamlet is a tragedy written by William Shakespeare around 1600 .",
    "p3": "The Globe Theatre staged many plays in London .",
}


def nq_item(*passage_ids: str, initial_answer: str = "Christopher Marlowe") -> DiagnosisInput:
    return DiagnosisInput(
        sample_id="s1",
        dataset=DatasetName.NQ,
        query="who wrote hamlet",
        evidence=[
            EvidenceItem(rank=rank, passage_id=passage_id, text=PASSAGES[passage_id], score=1.0 - 0.1 * rank)
            for rank, passage_id in enumerate(passage_ids, start=1)
        ],
        initial_answer=initial_answer,
    )


def fever_item(initial_answer: str = "REFUTES") -> DiagnosisInput:
    return DiagnosisInput(
        sample_id="42",
        dataset=DatasetName.FEVER,
        query="Hamlet was written by William Shakespeare .",
        evidence=[
            EvidenceItem(rank=1, passage_id="p1", text=PASSAGES["p1"], score=0.9),
            EvidenceItem(rank=2, passage_id="p2", text=PASSAGES["p2"], score=0.8),
        ],
        initial_answer=initial_answer,
    )


def decision(action: ControllerAction = ControllerAction.EVIDENCE_REFINEMENT) -> ControllerDecision:
    label = {
        ControllerAction.NO_ACTION: DiagnosisLabel.CORRECT,
        ControllerAction.RETRIEVAL_ENHANCEMENT: DiagnosisLabel.EVIDENCE_INSUFFICIENCY,
        ControllerAction.EVIDENCE_REFINEMENT: DiagnosisLabel.EVIDENCE_ACCESSIBILITY,
        ControllerAction.GENERATION_REPAIR: DiagnosisLabel.EVIDENCE_UTILISATION,
    }[action]
    return ControllerDecision(
        action=action,
        diagnosis_label=label,
        diagnosis_confidence=0.8,
        used_probe=False,
        probe_overrode_diagnosis=False,
        sample_id="s1",
    )


class Recorder:
    """Records every scorer and generator call and replays scripted answers."""

    def __init__(self, scores: dict[str, float], answer: str = "William Shakespeare"):
        self._scores = scores
        self._answer = answer
        self.scored: list[tuple[str, tuple[str, ...]]] = []
        self.prompts: list[str] = []

    def score(self, query, texts):
        self.scored.append((query, tuple(texts)))
        return [self._scores[text] for text in texts]

    def generate(self, prompt):
        self.prompts.append(prompt)
        return self._answer

    def prompt_builder(self, item, evidence):
        return "|".join(passage.passage_id for passage in evidence)


def recorder(answer: str = "William Shakespeare", **by_passage: float) -> Recorder:
    return Recorder({PASSAGES[passage_id]: score for passage_id, score in by_passage.items()}, answer)


def refiner_for(rec: Recorder, config: RefinementConfig = KEEP_ALL, candidate_provider=None) -> EvidenceRefiner:
    return EvidenceRefiner(
        rec.score, rec.generate, config=config, candidate_provider=candidate_provider, prompt_builder=rec.prompt_builder
    )


class GroundingTest(unittest.TestCase):
    def test_refusals(self):
        for answer in ("", "   ", "unknown", "Not specified", "N/A", "unanswerable", "none"):
            self.assertTrue(is_refusal(answer), answer)
        for answer in ("William Shakespeare", "1600", "no"):
            self.assertFalse(is_refusal(answer), answer)

    def test_grounding_ignores_case_punctuation_and_articles(self):
        evidence = nq_item("p2").evidence
        self.assertTrue(is_grounded("william shakespeare", evidence))
        self.assertTrue(is_grounded("William Shakespeare.", evidence))
        self.assertTrue(is_grounded("the Globe", nq_item("p3").evidence))

    def test_grounding_requires_a_contiguous_token_run(self):
        evidence = nq_item("p2").evidence
        self.assertFalse(is_grounded("Shakespeare Hamlet", evidence))
        self.assertFalse(is_grounded("Ben Jonson", evidence))
        self.assertFalse(is_grounded("", evidence))

    def test_rejection_reasons(self):
        item = nq_item("p1", "p2")
        self.assertIsNone(rejection_reason(item, "William Shakespeare", item.evidence))
        self.assertEqual(rejection_reason(item, "unknown", item.evidence), "refusal_or_empty")
        self.assertEqual(rejection_reason(item, "Ben Jonson", item.evidence), "not_grounded_in_kept_evidence")
        self.assertIsNone(
            rejection_reason(item, "Ben Jonson", item.evidence, RefinementConfig(require_grounded=False))
        )

    def test_fever_rejection_uses_the_label_parser(self):
        item = fever_item()
        self.assertIsNone(rejection_reason(item, "SUPPORTS", item.evidence))
        self.assertEqual(rejection_reason(item, "maybe", item.evidence), "unparsable_fever_label")
        self.assertEqual(rejection_reason(item, "", item.evidence), "refusal_or_empty")


class RoutingGateTest(unittest.TestCase):
    def test_other_actions_are_left_untouched(self):
        for action in (
            ControllerAction.NO_ACTION,
            ControllerAction.RETRIEVAL_ENHANCEMENT,
            ControllerAction.GENERATION_REPAIR,
        ):
            rec = recorder(p1=0.1, p2=0.9)
            item = nq_item("p1", "p2")
            result = refiner_for(rec).run(item, decision(action))
            self.assertEqual(result.status, RefinementStatus.NOT_ROUTED)
            self.assertEqual(result.final_answer, item.initial_answer)
            self.assertEqual(result.evidence, tuple(item.evidence))
            self.assertEqual(result.rerank_scores, ())
            self.assertFalse(result.regenerated)
            self.assertEqual(rec.scored, [])
            self.assertEqual(rec.prompts, [])

    def test_item_and_decision_types_are_enforced(self):
        rec = recorder(p1=0.1)
        refiner = refiner_for(rec)
        item = nq_item("p1")
        with self.assertRaises(TypeError):
            refiner.run("not an item", decision())
        with self.assertRaises(TypeError):
            refiner.run(item, "not a decision")
        with self.assertRaises(TypeError):
            refiner.run(item, decision(), probe_result="not a probe result")

    def test_constructor_validates_its_collaborators(self):
        rec = recorder(p1=0.1)
        with self.assertRaises(TypeError):
            EvidenceRefiner("not callable", rec.generate)
        with self.assertRaises(TypeError):
            EvidenceRefiner(rec.score, "not callable")
        with self.assertRaises(TypeError):
            EvidenceRefiner(rec.score, rec.generate, config={"keep_k": 3})
        with self.assertRaises(TypeError):
            EvidenceRefiner(rec.score, rec.generate, candidate_provider="not callable")
        with self.assertRaises(TypeError):
            EvidenceRefiner(rec.score, rec.generate, prompt_builder="not callable")


class RefinementOutcomeTest(unittest.TestCase):
    def test_unchanged_order_keeps_the_answer_without_generating(self):
        rec = recorder(p1=0.9, p2=0.5)
        item = nq_item("p1", "p2")
        result = refiner_for(rec).run(item, decision())
        self.assertEqual(result.status, RefinementStatus.NO_CHANGE)
        self.assertEqual(result.final_answer, item.initial_answer)
        self.assertFalse(result.regenerated)
        self.assertIsNone(result.candidate_answer)
        self.assertFalse(result.evidence_changed)
        self.assertEqual(rec.prompts, [])
        self.assertEqual(result.rerank_scores, (0.9, 0.5))

    def test_regenerate_when_unchanged_can_be_forced(self):
        rec = recorder(p1=0.9, p2=0.5)
        config = RefinementConfig(keep_k=5, regenerate_when_unchanged=True)
        result = refiner_for(rec, config).run(nq_item("p1", "p2"), decision())
        self.assertEqual(result.status, RefinementStatus.ANSWER_REPLACED)
        self.assertEqual(len(rec.prompts), 1)

    def test_grounded_answer_replaces_the_original(self):
        rec = recorder(p1=0.1, p2=0.9)
        item = nq_item("p1", "p2")
        result = refiner_for(rec).run(item, decision())
        self.assertEqual(result.status, RefinementStatus.ANSWER_REPLACED)
        self.assertEqual(result.final_answer, "William Shakespeare")
        self.assertEqual(result.candidate_answer, "William Shakespeare")
        self.assertTrue(result.answer_changed)
        self.assertTrue(result.evidence_changed)
        self.assertTrue(result.regenerated)
        self.assertIsNone(result.reject_reason)
        self.assertEqual(result.refined_passage_ids, ("p2", "p1"))
        self.assertEqual(result.original_passage_ids, ("p1", "p2"))
        self.assertEqual(result.sample_id, "s1")

    def test_ungrounded_answer_is_rejected(self):
        rec = recorder("Ben Jonson", p1=0.1, p2=0.9)
        result = refiner_for(rec).run(nq_item("p1", "p2"), decision())
        self.assertEqual(result.status, RefinementStatus.KEPT_ORIGINAL)
        self.assertEqual(result.final_answer, "Christopher Marlowe")
        self.assertEqual(result.reject_reason, "not_grounded_in_kept_evidence")
        self.assertTrue(result.regenerated)
        self.assertEqual(result.candidate_answer, "Ben Jonson")  # kept so offline analysis can audit the gate

    def test_refusal_is_rejected(self):
        rec = recorder("unknown", p1=0.1, p2=0.9)
        result = refiner_for(rec).run(nq_item("p1", "p2"), decision())
        self.assertEqual(result.status, RefinementStatus.KEPT_ORIGINAL)
        self.assertEqual(result.reject_reason, "refusal_or_empty")

    def test_answer_must_be_grounded_in_the_kept_passages_only(self):
        # p2 holds the answer but is pruned by keep_k=1, so the answer is no longer grounded.
        rec = recorder(p1=0.9, p2=0.5, p3=0.1)
        config = RefinementConfig(keep_k=1)
        result = refiner_for(rec, config).run(nq_item("p1", "p2", "p3"), decision())
        self.assertEqual(result.refined_passage_ids, ("p1",))
        self.assertEqual(result.dropped_passage_ids, ("p2", "p3"))
        self.assertEqual(result.status, RefinementStatus.KEPT_ORIGINAL)
        self.assertEqual(result.reject_reason, "not_grounded_in_kept_evidence")

    def test_pruning_reports_the_dropped_passages(self):
        rec = recorder(p1=0.1, p2=0.9, p3=0.5)
        result = refiner_for(rec, KEEP_TWO).run(nq_item("p1", "p2", "p3"), decision())
        self.assertEqual(result.refined_passage_ids, ("p2", "p3"))
        self.assertEqual(result.dropped_passage_ids, ("p1",))
        self.assertEqual(result.candidate_pool_size, 3)

    def test_answer_is_stripped(self):
        rec = recorder("  William Shakespeare \n", p1=0.1, p2=0.9)
        result = refiner_for(rec).run(nq_item("p1", "p2"), decision())
        self.assertEqual(result.final_answer, "William Shakespeare")

    def test_generator_must_return_text(self):
        rec = recorder(p1=0.1, p2=0.9)
        refiner = EvidenceRefiner(rec.score, lambda prompt: None, config=KEEP_ALL, prompt_builder=rec.prompt_builder)
        item = nq_item("p1", "p2")
        with self.assertRaises(TypeError):
            refiner.run(item, decision())

    def test_prompt_sees_only_the_kept_evidence_in_the_new_order(self):
        rec = recorder(p1=0.1, p2=0.9, p3=0.5)
        refiner_for(rec, KEEP_TWO).run(nq_item("p1", "p2", "p3"), decision())
        self.assertEqual(rec.prompts, ["p2|p3"])

    def test_fever_label_answer_is_accepted_without_grounding(self):
        rec = Recorder({PASSAGES["p1"]: 0.1, PASSAGES["p2"]: 0.9}, "SUPPORTS")
        result = refiner_for(rec).run(fever_item(), decision())
        self.assertEqual(result.status, RefinementStatus.ANSWER_REPLACED)
        self.assertEqual(result.final_answer, "SUPPORTS")

    def test_empty_evidence_is_a_no_change(self):
        rec = recorder()
        item = DiagnosisInput(
            sample_id="s1", dataset=DatasetName.NQ, query="who wrote hamlet", evidence=[], initial_answer="Marlowe"
        )
        result = refiner_for(rec).run(item, decision())
        self.assertEqual(result.status, RefinementStatus.NO_CHANGE)
        self.assertEqual(result.final_answer, "Marlowe")
        self.assertEqual(rec.scored, [])

    def test_probe_hints_are_passed_through_to_reranking(self):
        probe_result = ProbeResult.from_judgments("unsupported", "sufficient", "hard_to_access", (1,), (2,))
        rec = recorder(p1=0.9, p2=0.5)
        config = RefinementConfig(keep_k=5, probe_hint_weight=1.0)
        result = refiner_for(rec, config).run(nq_item("p1", "p2"), decision(), probe_result)
        self.assertTrue(result.used_probe_hint)
        self.assertEqual(result.refined_passage_ids, ("p2", "p1"))

    def test_candidate_provider_widens_the_pool(self):
        extra = [EvidenceItem(rank=1, passage_id="p3", text=PASSAGES["p3"], score=0.05)]
        rec = recorder(p1=0.1, p2=0.5, p3=0.9)
        result = refiner_for(rec, KEEP_TWO, candidate_provider=lambda _: extra).run(nq_item("p1", "p2"), decision())
        self.assertEqual(result.candidate_pool_size, 3)
        self.assertEqual(result.refined_passage_ids, ("p3", "p2"))
        self.assertEqual(result.dropped_passage_ids, ("p1",))

    def test_refinement_is_deterministic(self):
        first = refiner_for(recorder(p1=0.1, p2=0.9)).run(nq_item("p1", "p2"), decision())
        second = refiner_for(recorder(p1=0.1, p2=0.9)).run(nq_item("p1", "p2"), decision())
        self.assertEqual(first, second)


class IntegrationContractTest(unittest.TestCase):
    def test_calling_the_refiner_returns_the_final_answer(self):
        rec = recorder(p1=0.1, p2=0.9)
        item, routed = nq_item("p1", "p2"), decision()
        refiner = refiner_for(rec)
        self.assertEqual(refiner(item, routed), "William Shakespeare")
        self.assertEqual(evidence_refinement(item, routed, refiner), "William Shakespeare")

    def test_wrapper_validates_the_refiner(self):
        item, routed = nq_item("p1"), decision()
        with self.assertRaises(TypeError):
            evidence_refinement(item, routed, refiner="not a refiner")

    def test_generate_exposes_one_backend_call_for_offline_evaluation(self):
        rec = recorder("  William Shakespeare \n", p1=0.1, p2=0.9)
        refiner = refiner_for(rec)
        self.assertEqual(refiner.generate("a prompt"), "William Shakespeare")
        self.assertEqual(rec.prompts, ["a prompt"])

    def test_generate_rejects_a_non_text_backend_reply(self):
        rec = recorder(p1=0.1)
        refiner = EvidenceRefiner(rec.score, lambda prompt: 42, config=KEEP_ALL, prompt_builder=rec.prompt_builder)
        with self.assertRaises(TypeError):
            refiner.generate("a prompt")

    def test_run_returns_a_refinement_result(self):
        rec = recorder(p1=0.1, p2=0.9)
        self.assertIsInstance(refiner_for(rec).run(nq_item("p1", "p2"), decision()), RefinementResult)

    def test_default_prompt_builder_is_the_frozen_reference_rag_prompt(self):
        from src.generation.qwen import FEVER_PROMPT_INSTRUCTION, PROMPT_INSTRUCTION

        item = nq_item("p1", "p2")
        prompt = build_refinement_prompt(item, item.evidence[1:])
        self.assertTrue(prompt.startswith(PROMPT_INSTRUCTION))
        self.assertIn("[1] " + PASSAGES["p2"], prompt)
        self.assertNotIn(PASSAGES["p1"], prompt)
        self.assertIn("Question: who wrote hamlet", prompt)

        claim = fever_item()
        fever_prompt = build_refinement_prompt(claim, claim.evidence)
        self.assertTrue(fever_prompt.startswith(FEVER_PROMPT_INSTRUCTION))
        self.assertIn("Claim: " + claim.query, fever_prompt)


if __name__ == "__main__":
    unittest.main()
