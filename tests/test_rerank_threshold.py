"""Tests for the score-threshold sweep. No model is loaded."""

from __future__ import annotations

import unittest

from scripts.analyze_rerank_threshold import carries_answer, sweep


def at(threshold: float, rows: list[dict]) -> dict:
    return next(row for row in rows if row["threshold"] == threshold)


class CarriesAnswerTest(unittest.TestCase):
    def test_a_contiguous_token_run_counts(self):
        self.assertTrue(carries_answer("William Shakespeare", "written by William Shakespeare in 1600"))

    def test_normalisation_applies(self):
        self.assertTrue(carries_answer("the Globe.", "at The  Globe , London"))

    def test_a_scattered_match_does_not_count(self):
        self.assertFalse(carries_answer("Shakespeare Hamlet", "Hamlet by Shakespeare"))

    def test_an_empty_answer_never_matches(self):
        self.assertFalse(carries_answer("", "anything"))


class SweepTest(unittest.TestCase):
    def setUp(self):
        # one sample: answer at 0.6, distractors at 0.4 and 0.1
        self.samples = [[[0.6, True], [0.4, False], [0.1, False]]]

    def test_a_low_threshold_keeps_everything(self):
        row = at(0.0, sweep(self.samples, (0.0,)))
        self.assertEqual(row["answer_passages_kept"], 1.0)
        self.assertEqual(row["distractors_kept"], 1.0)
        self.assertEqual(row["mean_passages_per_sample"], 3.0)
        self.assertEqual(row["samples_that_lost_their_only_answer"], 0)

    def test_a_middle_threshold_drops_only_distractors(self):
        row = at(0.5, sweep(self.samples, (0.5,)))
        self.assertEqual(row["answer_passages_kept"], 1.0)
        self.assertEqual(row["distractors_kept"], 0.0)
        self.assertEqual(row["mean_passages_per_sample"], 1.0)
        self.assertEqual(row["samples_that_lost_their_only_answer"], 0)

    def test_a_high_threshold_loses_the_answer(self):
        row = at(0.9, sweep(self.samples, (0.9,)))
        self.assertEqual(row["answer_passages_kept"], 0.0)
        self.assertEqual(row["samples_that_lost_their_only_answer"], 1)
        self.assertEqual(row["samples_emptied"], 1)

    def test_a_sample_that_never_had_the_answer_cannot_lose_it(self):
        rows = sweep([[[0.4, False], [0.1, False]]], (0.9,))
        self.assertEqual(at(0.9, rows)["samples_that_lost_their_only_answer"], 0)
        self.assertEqual(at(0.9, rows)["samples_emptied"], 1)

    def test_retention_is_monotone_in_the_threshold(self):
        samples = [[[0.9, True], [0.5, True], [0.3, False], [0.2, False]]]
        rows = sweep(samples, (0.0, 0.25, 0.4, 0.6, 0.95))
        kept = [row["answer_passages_kept"] for row in rows]
        self.assertEqual(kept, sorted(kept, reverse=True))
        counts = [row["mean_passages_per_sample"] for row in rows]
        self.assertEqual(counts, sorted(counts, reverse=True))

    def test_rates_are_over_the_whole_set(self):
        samples = [[[0.9, True], [0.1, False]], [[0.2, True], [0.8, False]]]
        row = at(0.5, sweep(samples, (0.5,)))
        self.assertEqual(row["answer_passages_kept"], 0.5)
        self.assertEqual(row["distractors_kept"], 0.5)
        self.assertEqual(row["samples_that_lost_their_only_answer"], 1)

    def test_no_samples_gives_no_rates(self):
        row = at(0.5, sweep([], (0.5,)))
        self.assertIsNone(row["answer_passages_kept"])
        self.assertEqual(row["mean_passages_per_sample"], 0.0)


if __name__ == "__main__":
    unittest.main()


class RelativeSweepTest(unittest.TestCase):
    def setUp(self):
        from scripts.analyze_rerank_threshold import sweep_relative

        self.sweep_relative = sweep_relative
        # best is 0.6, so ratio r keeps everything scoring >= 0.6r
        self.samples = [[[0.6, True], [0.4, False], [0.1, False]]]

    def rows(self, *ratios):
        return {row["ratio"]: row for row in self.sweep_relative(self.samples, ratios)}

    def test_ratio_zero_keeps_everything(self):
        row = self.rows(0.0)[0.0]
        self.assertEqual(row["mean_passages_per_sample"], 3.0)
        self.assertEqual(row["distractors_kept"], 1.0)

    def test_a_middle_ratio_drops_the_weakest(self):
        row = self.rows(0.5)[0.5]  # cut at 0.30
        self.assertEqual(row["mean_passages_per_sample"], 2.0)
        self.assertEqual(row["answer_passages_kept"], 1.0)
        self.assertEqual(row["distractors_kept"], 0.5)

    def test_the_best_passage_always_survives(self):
        for ratio in (0.0, 0.5, 0.9, 1.0):
            row = self.rows(ratio)[ratio]
            self.assertGreaterEqual(row["mean_passages_per_sample"], 1.0, ratio)

    def test_it_adapts_to_each_sample_scale(self):
        # One sample scores high overall, the other low. Ratio 0.6 cuts the first
        # at 0.54 and the second at 0.12, so each keeps its answer and drops its
        # own distractor -- an absolute cut could not do both.
        samples = [[[0.9, True], [0.5, False]], [[0.2, True], [0.1, False]]]
        rows = {r["ratio"]: r for r in self.sweep_relative(samples, (0.6,))}
        self.assertEqual(rows[0.6]["answer_passages_kept"], 1.0)
        self.assertEqual(rows[0.6]["distractors_kept"], 0.0)
        self.assertEqual(rows[0.6]["mean_passages_per_sample"], 1.0)

    def test_an_absolute_cut_cannot_do_the_same(self):
        from scripts.analyze_rerank_threshold import sweep

        samples = [[[0.9, True], [0.5, False]], [[0.2, True], [0.1, False]]]
        # Any absolute cut either keeps the first sample's distractor or loses
        # the second sample's answer.
        keeps_distractor = sweep(samples, (0.15,))[0]
        loses_answer = sweep(samples, (0.55,))[0]
        self.assertEqual(keeps_distractor["distractors_kept"], 0.5)
        self.assertEqual(loses_answer["samples_that_lost_their_only_answer"], 1)

    def test_a_sample_whose_answer_is_not_the_best_can_lose_it(self):
        samples = [[[0.9, False], [0.2, True]]]
        rows = {r["ratio"]: r for r in self.sweep_relative(samples, (0.5,))}
        self.assertEqual(rows[0.5]["samples_that_lost_their_only_answer"], 1)

    def test_an_empty_sample_is_tolerated(self):
        rows = {r["ratio"]: r for r in self.sweep_relative([[]], (0.5,))}
        self.assertEqual(rows[0.5]["mean_passages_per_sample"], 0.0)
