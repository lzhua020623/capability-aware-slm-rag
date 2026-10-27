"""Tests for the evidence refinement config and result schema. No model is loaded."""

from __future__ import annotations

import dataclasses
import math
import unittest

from src.compensation.refinement_schema import (
    DEFAULT_REFINEMENT_CONFIG,
    RefinementConfig,
    RefinementResult,
    RefinementStatus,
    assert_no_forbidden_fields,
)
from src.diagnosis.schema import FORBIDDEN_RUNTIME_FIELDS, EvidenceItem
from src.probe.schema import MAX_EVIDENCE_RANK


def evidence(*specs: tuple[int, str]) -> tuple[EvidenceItem, ...]:
    return tuple(
        EvidenceItem(rank=rank, passage_id=passage_id, text=f"text {passage_id}", score=1.0 / rank)
        for rank, passage_id in specs
    )


def result(**overrides) -> RefinementResult:
    kept = overrides.pop("evidence", evidence((1, "p1"), (2, "p2")))
    fields = {
        "status": RefinementStatus.NO_CHANGE,
        "initial_answer": "Paris",
        "final_answer": "Paris",
        "evidence": kept,
        "rerank_scores": tuple(float(len(kept) - i) for i in range(len(kept))),
        "original_passage_ids": tuple(item.passage_id for item in kept),
        "dropped_passage_ids": (),
        "candidate_pool_size": len(kept),
        "regenerated": False,
        "used_probe_hint": False,
    }
    fields.update(overrides)
    return RefinementResult(**fields)


class RefinementConfigTest(unittest.TestCase):
    def test_defaults_are_conservative(self):
        config = DEFAULT_REFINEMENT_CONFIG
        self.assertEqual(config.keep_k, MAX_EVIDENCE_RANK)  # reorder only; pruning is opt-in
        self.assertIsNone(config.min_score)
        self.assertEqual(config.probe_hint_weight, 0.0)
        self.assertTrue(config.require_grounded)
        self.assertFalse(config.regenerate_when_unchanged)
        self.assertEqual(config.reranker_model, "BAAI/bge-reranker-large")

    def test_keep_k_bounds(self):
        for keep_k in (1, MAX_EVIDENCE_RANK):
            self.assertEqual(RefinementConfig(keep_k=keep_k).keep_k, keep_k)
        for keep_k in (0, -1, MAX_EVIDENCE_RANK + 1):
            with self.assertRaises(ValueError):
                RefinementConfig(keep_k=keep_k)

    def test_keep_k_type(self):
        for keep_k in (True, 2.0, "3"):
            with self.assertRaises(TypeError):
                RefinementConfig(keep_k=keep_k)

    def test_min_score_must_be_finite_or_none(self):
        self.assertEqual(RefinementConfig(min_score=-2).min_score, -2.0)
        self.assertIsNone(RefinementConfig().min_score)
        for value in (math.nan, math.inf):
            with self.assertRaises(ValueError):
                RefinementConfig(min_score=value)
        with self.assertRaises(TypeError):
            RefinementConfig(min_score="0.0")

    def test_probe_hint_weight_validated(self):
        self.assertEqual(RefinementConfig(probe_hint_weight=1.5).probe_hint_weight, 1.5)
        with self.assertRaises(ValueError):
            RefinementConfig(probe_hint_weight=math.nan)
        with self.assertRaises(TypeError):
            RefinementConfig(probe_hint_weight=True)

    def test_flags_and_model_name(self):
        for name in ("require_grounded", "regenerate_when_unchanged"):
            with self.assertRaises(TypeError):
                RefinementConfig(**{name: 1})
        for model in ("", "   "):
            with self.assertRaises(ValueError):
                RefinementConfig(reranker_model=model)

    def test_config_is_frozen_and_serialisable(self):
        with self.assertRaises(dataclasses.FrozenInstanceError):
            DEFAULT_REFINEMENT_CONFIG.keep_k = 5  # type: ignore[misc]
        self.assertEqual(
            set(DEFAULT_REFINEMENT_CONFIG.to_dict()),
            {"keep_k", "min_score", "probe_hint_weight", "require_grounded", "regenerate_when_unchanged", "reranker_model"},
        )


class RefinementResultTest(unittest.TestCase):
    def test_derived_fields(self):
        kept = evidence((1, "p2"), (2, "p1"))
        refined = result(
            status=RefinementStatus.ANSWER_REPLACED,
            evidence=kept,
            final_answer="Lyon",
            candidate_answer="Lyon",
            original_passage_ids=("p1", "p2"),
            regenerated=True,
        )
        self.assertEqual(refined.refined_passage_ids, ("p2", "p1"))
        self.assertTrue(refined.answer_changed)
        self.assertTrue(refined.evidence_changed)

    def test_unchanged_evidence_and_answer(self):
        refined = result()
        self.assertFalse(refined.answer_changed)
        self.assertFalse(refined.evidence_changed)

    def test_rerank_scores_may_be_empty_but_otherwise_must_align(self):
        self.assertEqual(result(rerank_scores=()).rerank_scores, ())
        with self.assertRaises(ValueError):
            result(rerank_scores=(1.0,))
        with self.assertRaises(ValueError):
            result(rerank_scores=(1.0, math.nan))

    def test_candidate_pool_must_cover_kept_evidence(self):
        with self.assertRaises(ValueError):
            result(candidate_pool_size=1)

    def test_candidate_answer_requires_a_generation_call(self):
        with self.assertRaises(ValueError):
            result(candidate_answer="Lyon")
        with self.assertRaises(TypeError):
            result(status=RefinementStatus.KEPT_ORIGINAL, regenerated=True, reject_reason="r", candidate_answer=7)

    def test_rejected_candidate_is_kept_for_analysis(self):
        refined = result(
            status=RefinementStatus.KEPT_ORIGINAL,
            regenerated=True,
            reject_reason="not_grounded_in_kept_evidence",
            candidate_answer="Lyon",
        )
        self.assertEqual(refined.candidate_answer, "Lyon")
        self.assertEqual(refined.final_answer, "Paris")
        self.assertFalse(refined.answer_changed)

    def test_accepted_candidate_must_be_the_final_answer(self):
        with self.assertRaises(ValueError):
            result(
                status=RefinementStatus.ANSWER_REPLACED,
                regenerated=True,
                final_answer="Lyon",
                candidate_answer="Nice",
            )

    def test_status_invariants(self):
        with self.assertRaises(ValueError):  # a kept-answer status cannot change the answer
            result(status=RefinementStatus.NO_CHANGE, final_answer="Lyon")
        with self.assertRaises(ValueError):  # not_routed cannot reorder evidence
            result(status=RefinementStatus.NOT_ROUTED, original_passage_ids=("p2", "p1"))
        with self.assertRaises(ValueError):  # not_routed cannot regenerate
            result(status=RefinementStatus.NOT_ROUTED, regenerated=True)
        with self.assertRaises(ValueError):  # no_change must not spend a generation call
            result(status=RefinementStatus.NO_CHANGE, regenerated=True)
        with self.assertRaises(ValueError):  # kept_original means the generator did run
            result(status=RefinementStatus.KEPT_ORIGINAL, reject_reason="not_grounded_in_kept_evidence")
        with self.assertRaises(ValueError):  # kept_original needs a reason
            result(status=RefinementStatus.KEPT_ORIGINAL, regenerated=True)
        with self.assertRaises(ValueError):  # answer_replaced must not carry a reason
            result(status=RefinementStatus.ANSWER_REPLACED, regenerated=True, reject_reason="refusal_or_empty")

    def test_valid_statuses(self):
        self.assertEqual(result(status=RefinementStatus.NOT_ROUTED).status, RefinementStatus.NOT_ROUTED)
        kept_original = result(
            status=RefinementStatus.KEPT_ORIGINAL,
            regenerated=True,
            reject_reason="not_grounded_in_kept_evidence",
            candidate_answer="Lyon",
        )
        self.assertTrue(kept_original.regenerated)
        self.assertFalse(kept_original.answer_changed)

    def test_type_checks(self):
        with self.assertRaises(TypeError):
            result(initial_answer=None)
        with self.assertRaises(TypeError):
            RefinementResult(
                status=RefinementStatus.NO_CHANGE,
                initial_answer="Paris",
                final_answer="Paris",
                evidence=("p1",),
                rerank_scores=(),
                original_passage_ids=("p1",),
                dropped_passage_ids=(),
                candidate_pool_size=1,
                regenerated=False,
                used_probe_hint=False,
            )
        with self.assertRaises(TypeError):
            result(candidate_pool_size=2.0)
        with self.assertRaises(TypeError):
            result(regenerated=1)
        with self.assertRaises(TypeError):
            result(reject_reason=7)
        with self.assertRaises(ValueError):
            result(dropped_passage_ids=("",))

    def test_sample_id_is_logging_only(self):
        self.assertEqual(result(sample_id="a"), result(sample_id="b"))

    def test_result_is_frozen(self):
        with self.assertRaises(dataclasses.FrozenInstanceError):
            result().final_answer = "Lyon"  # type: ignore[misc]


class ForbiddenFieldTest(unittest.TestCase):
    def test_no_offline_only_fields_in_runtime_objects(self):
        for cls in (RefinementConfig, RefinementResult):
            assert_no_forbidden_fields(cls.__dataclass_fields__)
        assert_no_forbidden_fields(result().to_dict())
        assert_no_forbidden_fields(DEFAULT_REFINEMENT_CONFIG.to_dict())

    def test_guard_rejects_forbidden_names(self):
        for name in sorted(FORBIDDEN_RUNTIME_FIELDS):
            with self.assertRaises(ValueError):
                assert_no_forbidden_fields({name, "status"})

    def test_result_dict_shape(self):
        keys = set(result().to_dict())
        self.assertEqual(
            keys,
            {
                "sample_id",
                "status",
                "initial_answer",
                "final_answer",
                "answer_changed",
                "evidence_changed",
                "original_passage_ids",
                "refined_passage_ids",
                "dropped_passage_ids",
                "rerank_scores",
                "candidate_pool_size",
                "regenerated",
                "used_probe_hint",
                "reject_reason",
                "candidate_answer",
            },
        )


if __name__ == "__main__":
    unittest.main()
