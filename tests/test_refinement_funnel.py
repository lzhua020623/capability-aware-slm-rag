"""Tests for the repair funnel analysis. No model is loaded."""

from __future__ import annotations

import unittest

from scripts.analyze_refinement_funnel import answer_bearing_ids, bucket, build_funnel, classify, rate
from src.diagnosis.schema import EvidenceItem


def passage(rank: int, text: str) -> EvidenceItem:
    return EvidenceItem(rank=rank, passage_id=f"p{rank}", text=text, score=1.0 - 0.1 * rank)


def row(before: list[str], after: list[str], bearing: set[str], em_before: int = 0, em_after: int = 0) -> dict:
    data = {
        "before_ids": before,
        "after_ids": after,
        "answer_bearing_ids": bearing,
        "em_before": em_before,
        "em_after": em_after,
    }
    data.update(classify(data))
    return data


class AnswerBearingTest(unittest.TestCase):
    def test_finds_every_passage_holding_the_answer(self):
        evidence = [
            passage(1, "Christopher Marlowe died in 1593 ."),
            passage(2, "Hamlet was written by William Shakespeare ."),
            passage(3, "William Shakespeare also wrote Macbeth ."),
        ]
        self.assertEqual(answer_bearing_ids("William Shakespeare", evidence), {"p2", "p3"})

    def test_matching_ignores_case_and_punctuation(self):
        self.assertEqual(answer_bearing_ids("the globe", [passage(1, "The Globe , in London .")]), {"p1"})

    def test_an_empty_gold_answer_matches_nothing(self):
        self.assertEqual(answer_bearing_ids("", [passage(1, "anything")]), set())

    def test_a_non_contiguous_match_does_not_count(self):
        self.assertEqual(answer_bearing_ids("Shakespeare Hamlet", [passage(1, "Hamlet by Shakespeare")]), set())


class ClassifyTest(unittest.TestCase):
    def test_promotion_is_detected(self):
        flags = classify({"before_ids": ["p1", "p2"], "after_ids": ["p2", "p1"], "answer_bearing_ids": {"p2"}})
        self.assertTrue(flags["has_answer_passage"])
        self.assertTrue(flags["rank1_changed"])
        self.assertFalse(flags["answer_at_rank1_before"])
        self.assertTrue(flags["answer_at_rank1_after"])

    def test_demotion_is_detected(self):
        flags = classify({"before_ids": ["p1", "p2"], "after_ids": ["p2", "p1"], "answer_bearing_ids": {"p1"}})
        self.assertTrue(flags["answer_at_rank1_before"])
        self.assertFalse(flags["answer_at_rank1_after"])

    def test_an_unchanged_top_is_not_a_rank1_change(self):
        flags = classify({"before_ids": ["p1", "p2"], "after_ids": ["p1", "p3"], "answer_bearing_ids": {"p1"}})
        self.assertFalse(flags["rank1_changed"])

    def test_no_answer_passage(self):
        flags = classify({"before_ids": ["p1"], "after_ids": ["p1"], "answer_bearing_ids": set()})
        self.assertFalse(flags["has_answer_passage"])
        self.assertFalse(flags["answer_at_rank1_after"])

    def test_empty_evidence_is_handled(self):
        flags = classify({"before_ids": [], "after_ids": [], "answer_bearing_ids": set()})
        self.assertFalse(flags["rank1_changed"])


class BucketTest(unittest.TestCase):
    def test_rate_counts_only_already_wrong_samples(self):
        rows = [
            row(["p1"], ["p1"], {"p1"}, em_before=0, em_after=1),
            row(["p1"], ["p1"], {"p1"}, em_before=0, em_after=0),
            row(["p1"], ["p1"], {"p1"}, em_before=1, em_after=1),
        ]
        stats = bucket(rows, "x")
        self.assertEqual(stats["samples"], 3)
        self.assertEqual(stats["already_wrong"], 2)
        self.assertEqual(stats["repaired"], 1)
        self.assertEqual(stats["repair_rate"], 0.5)

    def test_an_empty_bucket_has_no_rate(self):
        self.assertIsNone(bucket([], "empty")["repair_rate"])
        self.assertIsNone(rate(0, 0))


class FunnelTest(unittest.TestCase):
    def test_stages_partition_the_samples(self):
        rows = [
            row(["p1"], ["p1"], set()),                       # no answer passage
            row(["p1", "p2"], ["p1", "p2"], {"p1"}),          # already first, held
            row(["p1", "p2"], ["p2", "p1"], {"p1"}),          # already first, demoted
            row(["p1", "p2"], ["p2", "p1"], {"p2"}),          # promoted
            row(["p1", "p2"], ["p1", "p2"], {"p2"}),          # present, not surfaced
        ]
        funnel = {stage["stage"][:2]: stage for stage in build_funnel(rows)}
        self.assertEqual(funnel["al"]["samples"], 5)
        self.assertEqual(funnel["1."]["samples"], 1)
        self.assertEqual(funnel["2."]["samples"], 2)
        self.assertEqual(funnel["2a"]["samples"], 1)
        self.assertEqual(funnel["2b"]["samples"], 1)
        self.assertEqual(funnel["3."]["samples"], 1)
        self.assertEqual(funnel["4."]["samples"], 1)

    def test_the_four_top_level_stages_sum_to_the_whole(self):
        rows = [
            row(["p1"], ["p1"], set()),
            row(["p1", "p2"], ["p1", "p2"], {"p1"}),
            row(["p1", "p2"], ["p2", "p1"], {"p2"}),
            row(["p1", "p2"], ["p1", "p2"], {"p2"}),
        ]
        stages = {stage["stage"][:2]: stage["samples"] for stage in build_funnel(rows)}
        self.assertEqual(stages["1."] + stages["2."] + stages["3."] + stages["4."], stages["al"])

    def test_perturbation_buckets_partition_the_samples(self):
        rows = [
            row(["p1", "p2"], ["p1", "p2"], {"p1"}),   # top unchanged
            row(["p1", "p2"], ["p2", "p1"], {"p2"}),   # top changed
            row(["p1", "p2"], ["p2", "p1"], {"p1"}),   # top changed
        ]
        stages = {stage["stage"][:2]: stage["samples"] for stage in build_funnel(rows)}
        self.assertEqual(stages["A."], 1)
        self.assertEqual(stages["B."], 2)
        self.assertEqual(stages["A."] + stages["B."], stages["al"])

    def test_substages_sum_to_their_parent(self):
        rows = [
            row(["p1", "p2"], ["p1", "p2"], {"p1"}),
            row(["p1", "p2"], ["p2", "p1"], {"p1"}),
            row(["p1", "p2"], ["p2", "p1"], {"p1"}),
        ]
        stages = {stage["stage"][:2]: stage["samples"] for stage in build_funnel(rows)}
        self.assertEqual(stages["2a"] + stages["2b"], stages["2."])


if __name__ == "__main__":
    unittest.main()
