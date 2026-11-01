"""Unit tests for unified conservative normalized scoring (no model inference)."""

from __future__ import annotations

import unittest

from baselines._lib.answer_scoring import (
    normalize_form,
    normalize_standard,
    score_answer,
)
from src.evaluation.qa import best_em_f1, exact_match


class NormalizeFormTests(unittest.TestCase):
    def test_case(self):
        self.assertEqual(normalize_form("Blue"), normalize_form("blue"))

    def test_punctuation(self):
        self.assertEqual(normalize_form("blue."), normalize_form("blue"))

    def test_article(self):
        self.assertEqual(normalize_form("the sky"), normalize_form("sky"))

    def test_multiple_spaces(self):
        self.assertEqual(normalize_form("a   b"), normalize_form("a b"))

    def test_hyphen_vs_space(self):
        self.assertEqual(normalize_form("well-known"), normalize_form("well known"))

    def test_unicode_separator(self):
        self.assertEqual(normalize_form("well\u2013known"), normalize_form("well known"))

    def test_semantic_error_not_matched(self):
        self.assertNotEqual(normalize_form("paris"), normalize_form("london"))
        s = score_answer("paris", ["london"])
        self.assertEqual(s["em"], 0)
        self.assertEqual(s["normalized_match"], 0)
        self.assertEqual(s["normalized_match_type"], "none")
        self.assertNotIn("relaxed_match", s)


class MultiGoldAndRegressionTests(unittest.TestCase):
    def test_multi_gold_best(self):
        s = score_answer("NYC", ["New York", "NYC", "New York City"])
        self.assertEqual(s["em"], 1)
        self.assertEqual(s["normalized_match"], 1)
        self.assertEqual(s["normalized_match_type"], "exact")

    def test_em_f1_regression_matches_qa_module(self):
        cases = [
            ("Blue", ["blue"]),
            ("the answer.", ["answer"]),
            ("foo", ["bar"]),
            ("well known", ["well-known"]),
        ]
        for pred, golds in cases:
            em, f1 = best_em_f1(pred, golds)
            scored = score_answer(pred, golds)
            self.assertEqual(scored["em"], em, msg=(pred, golds))
            self.assertAlmostEqual(scored["f1"], f1, places=9, msg=(pred, golds))

    def test_separator_type_when_em_fails(self):
        pred, gold = "well-known", ["well known"]
        self.assertEqual(exact_match(pred, gold[0]), 0)
        scored = score_answer(pred, gold)
        self.assertEqual(scored["em"], 0)
        self.assertEqual(scored["normalized_match"], 1)
        self.assertEqual(scored["normalized_match_type"], "separator_normalization")
        # standard path alone would not equate (hyphen stripped vs space kept differently)
        self.assertNotEqual(normalize_standard(pred), normalize_standard(gold[0]))

    def test_standard_normalization_type(self):
        scored = score_answer("The Answer.", ["answer"])
        self.assertEqual(scored["normalized_match"], 1)
        self.assertIn(
            scored["normalized_match_type"],
            {"standard_normalization", "exact", "spacing_normalization"},
        )

    def test_score_field_keys_no_relaxed(self):
        s = score_answer("a", ["a"])
        self.assertEqual(
            set(s),
            {"em", "f1", "normalized_match", "normalized_match_type"},
        )


if __name__ == "__main__":
    unittest.main()
