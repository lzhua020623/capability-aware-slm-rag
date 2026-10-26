"""Tests for the answer-matching sensitivity analysis. No model is loaded."""

from __future__ import annotations

import unittest

from scripts.analyze_answer_matching import (
    MATCHERS,
    contains_either_way,
    is_strict_substring,
    rate,
    score_run,
    span_report,
)


def row(initial: str, final: str, gold: str) -> dict:
    return {"initial_answer": initial, "final_answer": final, "gold_answer": gold}


class ContainmentTest(unittest.TestCase):
    def test_a_shorter_span_of_the_gold_counts(self):
        self.assertTrue(contains_either_way("Atlanta, Georgia", "at Mercedes-Benz Stadium in Atlanta, Georgia"))

    def test_a_longer_answer_than_the_gold_counts_too(self):
        self.assertTrue(contains_either_way("the city of Atlanta Georgia", "Atlanta Georgia"))

    def test_unrelated_strings_do_not(self):
        self.assertFalse(contains_either_way("Boston", "Atlanta Georgia"))

    def test_empty_strings_never_match(self):
        self.assertFalse(contains_either_way("", "Atlanta"))
        self.assertFalse(contains_either_way("Atlanta", ""))

    def test_normalisation_applies(self):
        self.assertTrue(contains_either_way("the Globe.", "The  Globe Theatre"))


class StrictSubstringTest(unittest.TestCase):
    def test_only_the_shorter_direction_counts(self):
        self.assertTrue(is_strict_substring("Cosmo Gordon Lang", "Cosmo Gordon Lang, Archbishop of Canterbury"))
        self.assertFalse(is_strict_substring("Cosmo Gordon Lang, Archbishop of Canterbury", "Cosmo Gordon Lang"))

    def test_an_identical_answer_is_not_a_shorter_span(self):
        self.assertFalse(is_strict_substring("Atlanta Georgia", "Atlanta, Georgia"))

    def test_an_empty_answer_is_not_a_span(self):
        self.assertFalse(is_strict_substring("", "Atlanta"))


class MatchersTest(unittest.TestCase):
    def test_every_rule_is_at_least_as_lax_as_exact_match(self):
        pairs = [
            ("Atlanta, Georgia", "at Mercedes-Benz Stadium in Atlanta, Georgia"),
            ("William Shakespeare", "William Shakespeare"),
            ("Ben Jonson", "William Shakespeare"),
        ]
        for answer, gold in pairs:
            strict = MATCHERS["exact_match (frozen)"](answer, gold)
            for name, matcher in MATCHERS.items():
                if name != "exact_match (frozen)":
                    self.assertGreaterEqual(matcher(answer, gold), strict, f"{name} on {answer!r}")

    def test_containment_accepts_the_shortest_span_case(self):
        self.assertEqual(MATCHERS["containment"]("Atlanta Georgia", "in Atlanta Georgia today"), 1)
        self.assertEqual(MATCHERS["exact_match (frozen)"]("Atlanta Georgia", "in Atlanta Georgia today"), 0)


class ScoreRunTest(unittest.TestCase):
    def test_counts_repairs_and_breaks_per_rule(self):
        rows = [
            row("Lyon", "Paris", "Paris"),      # repaired under every rule
            row("Paris", "Lyon", "Paris"),      # broken under every rule
            row("Atlanta", "Atlanta", "in Atlanta today"),  # only the lax rules accept it
        ]
        result = score_run(rows)
        self.assertEqual(result["exact_match (frozen)"]["net"], 0)
        self.assertEqual(result["exact_match (frozen)"]["before"], 0.3333)
        self.assertEqual(result["containment"]["before"], 0.6667)
        self.assertEqual(result["containment"]["net"], 0)

    def test_every_declared_rule_is_reported(self):
        self.assertEqual(set(score_run([row("a", "a", "a")])), set(MATCHERS))


class SpanReportTest(unittest.TestCase):
    def test_it_separates_contained_from_shorter(self):
        answers = ["Atlanta", "in Atlanta today and more", "Boston", "Paris"]
        golds = ["in Atlanta today", "in Atlanta today", "in Atlanta today", "Paris"]
        report = span_report(answers, golds)
        self.assertEqual(report["answers"], 4)
        self.assertEqual(report["exact_match_wrong"], 3)
        self.assertEqual(report["wrong_but_contained"], 2)
        self.assertEqual(report["wrong_but_shorter_span_of_gold"], 1)

    def test_lengths_are_reported(self):
        # normalize_answer drops articles, so the fixture avoids a/an/the.
        report = span_report(["one two"], ["one two three four"])
        self.assertEqual(report["gold_words_median"], 4)
        self.assertEqual(report["answer_words_median"], 2)

    def test_empty_input_does_not_divide_by_zero(self):
        report = span_report([], [])
        self.assertIsNone(report["wrong_but_contained_rate_of_wrong"])
        self.assertIsNone(report["gold_words_median"])

    def test_rate_helper(self):
        self.assertEqual(rate(1, 4), 0.25)
        self.assertIsNone(rate(0, 0))


if __name__ == "__main__":
    unittest.main()
