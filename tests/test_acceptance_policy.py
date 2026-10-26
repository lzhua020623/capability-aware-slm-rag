"""Tests for the counterfactual acceptance-policy analysis. No model is loaded."""

from __future__ import annotations

import unittest

from scripts.analyze_acceptance_policy import POLICIES, decide, decompose_by_initial_correctness


def row(
    initial: str = "Christopher Marlowe",
    candidate: str | None = "William Shakespeare",
    candidate_grounded: bool = True,
    initial_grounded: bool = False,
) -> dict:
    return {
        "initial_answer": initial,
        "candidate_answer": candidate,
        "candidate_grounded": candidate_grounded,
        "initial_grounded": initial_grounded,
    }


class DecideTest(unittest.TestCase):
    def test_always_keep_never_replaces(self):
        self.assertEqual(decide("always_keep", row()), "Christopher Marlowe")

    def test_always_accept_ignores_every_guard(self):
        self.assertEqual(decide("always_accept", row(candidate_grounded=False)), "William Shakespeare")
        self.assertEqual(decide("always_accept", row(candidate="unknown")), "unknown")

    def test_grounded_accepts_a_grounded_candidate(self):
        self.assertEqual(decide("grounded", row()), "William Shakespeare")

    def test_grounded_rejects_an_ungrounded_candidate(self):
        self.assertEqual(decide("grounded", row(candidate_grounded=False)), "Christopher Marlowe")

    def test_grounded_rejects_a_refusal_even_when_marked_grounded(self):
        self.assertEqual(decide("grounded", row(candidate="unknown")), "Christopher Marlowe")

    def test_protect_keeps_an_original_that_the_evidence_supports(self):
        self.assertEqual(decide("protect_grounded_original", row(initial_grounded=True)), "Christopher Marlowe")

    def test_protect_still_replaces_an_unsupported_original(self):
        self.assertEqual(decide("protect_grounded_original", row(initial_grounded=False)), "William Shakespeare")

    def test_every_policy_keeps_the_original_when_no_candidate_exists(self):
        for policy in POLICIES:
            self.assertEqual(decide(policy, row(candidate=None)), "Christopher Marlowe", policy)

    def test_protect_is_never_more_aggressive_than_grounded(self):
        cases = [
            row(),
            row(initial_grounded=True),
            row(candidate_grounded=False),
            row(candidate_grounded=False, initial_grounded=True),
            row(candidate="unknown", initial_grounded=True),
        ]
        for case in cases:
            grounded = decide("grounded", case)
            protect = decide("protect_grounded_original", case)
            if protect != case["initial_answer"]:
                self.assertEqual(protect, grounded)

    def test_policy_names_are_the_declared_ones(self):
        self.assertEqual(
            set(POLICIES), {"always_keep", "always_accept", "grounded", "protect_grounded_original"}
        )


def outcome(before: int, after: int) -> dict:
    return {"em_before": before, "em_after": after}


class DecompositionTest(unittest.TestCase):
    def test_counts_and_rates(self):
        rows = [outcome(1, 0), outcome(1, 1), outcome(0, 1), outcome(0, 0), outcome(0, 0)]
        stats = decompose_by_initial_correctness(rows)
        self.assertEqual(stats["already_right"], 2)
        self.assertEqual(stats["broken"], 1)
        self.assertEqual(stats["break_rate"], 0.5)
        self.assertEqual(stats["already_wrong"], 3)
        self.assertEqual(stats["repaired"], 1)
        self.assertAlmostEqual(stats["repair_rate"], 1 / 3, places=4)

    def test_break_even_is_break_rate_over_repair_rate(self):
        # 20% of correct answers broken, 10% of wrong ones repaired -> need 2 wrong per right.
        rows = [outcome(1, 0)] + [outcome(1, 1)] * 4 + [outcome(0, 1)] + [outcome(0, 0)] * 9
        stats = decompose_by_initial_correctness(rows)
        self.assertEqual(stats["break_rate"], 0.2)
        self.assertEqual(stats["repair_rate"], 0.1)
        self.assertEqual(stats["break_even_wrong_per_right"], 2.0)

    def test_no_repairs_means_no_break_even_ratio(self):
        self.assertIsNone(decompose_by_initial_correctness([outcome(1, 0), outcome(0, 0)])["break_even_wrong_per_right"])

    def test_empty_groups_do_not_divide_by_zero(self):
        self.assertEqual(decompose_by_initial_correctness([])["break_rate"], 0.0)
        self.assertEqual(decompose_by_initial_correctness([outcome(0, 0)])["break_rate"], 0.0)

    def test_a_module_that_never_changes_anything_is_neutral(self):
        rows = [outcome(1, 1)] * 5 + [outcome(0, 0)] * 5
        stats = decompose_by_initial_correctness(rows)
        self.assertEqual(stats["broken"], 0)
        self.assertEqual(stats["repaired"], 0)


if __name__ == "__main__":
    unittest.main()
