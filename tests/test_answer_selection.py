"""Tests for the answer-selection analysis. No model is loaded."""

from __future__ import annotations

import unittest

from scripts.analyze_answer_selection import best_candidate, majority_vote, score


class MajorityVoteTest(unittest.TestCase):
    def test_a_repeated_answer_wins(self):
        self.assertEqual(majority_vote(["Paris", "Lyon", "Paris"], "Nice"), "Paris")

    def test_normalisation_does_not_split_a_majority(self):
        self.assertEqual(majority_vote(["the Globe .", "Globe", "the globe"], "x"), "the Globe .")

    def test_all_different_falls_back(self):
        self.assertEqual(majority_vote(["Paris", "Lyon", "Nice"], "fallback"), "fallback")

    def test_no_candidates_falls_back(self):
        self.assertEqual(majority_vote([], "fallback"), "fallback")

    def test_the_original_spelling_is_returned(self):
        self.assertEqual(majority_vote(["Paris.", "paris"], "x"), "Paris.")

    def test_a_tie_picks_the_first_of_the_tied_answers(self):
        # Counter.most_common keeps insertion order among equal counts.
        self.assertEqual(majority_vote(["Paris", "Lyon", "Paris", "Lyon"], "x"), "Paris")


class BestCandidateTest(unittest.TestCase):
    def test_it_finds_a_correct_candidate(self):
        self.assertEqual(best_candidate(["Lyon", "Paris"], "paris"), "Paris")

    def test_it_falls_back_to_the_first_when_none_is_right(self):
        self.assertEqual(best_candidate(["Lyon", "Nice"], "Paris"), "Lyon")

    def test_it_matches_up_to_normalisation(self):
        self.assertEqual(best_candidate(["the Paris ."], "Paris"), "the Paris .")


class ScoreTest(unittest.TestCase):
    def test_exact_match_rate(self):
        gold = {"a": "Paris", "b": "Lyon", "c": "Nice"}
        self.assertEqual(score({"a": "Paris", "b": "Paris", "c": "nice"}, gold), 0.6667)

    def test_empty_is_zero_not_an_error(self):
        self.assertEqual(score({}, {}), 0.0)


class OrderingPropertyTest(unittest.TestCase):
    def test_the_oracle_is_never_worse_than_any_single_candidate(self):
        gold = {"a": "Paris", "b": "Lyon"}
        first = {"a": "Paris", "b": "Nice"}
        second = {"a": "Nice", "b": "Lyon"}
        oracle = {key: best_candidate([first[key], second[key]], gold[key]) for key in gold}
        self.assertGreaterEqual(score(oracle, gold), max(score(first, gold), score(second, gold)))

    def test_voting_can_be_worse_than_the_best_arm(self):
        # Two arms agree on a wrong answer, the third is right: the vote loses it.
        gold = {"a": "Paris"}
        candidates = ["Lyon", "Lyon", "Paris"]
        voted = {"a": majority_vote(candidates, "Nice")}
        self.assertEqual(score(voted, gold), 0.0)


if __name__ == "__main__":
    unittest.main()
