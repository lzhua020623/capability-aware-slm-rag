"""Tests for the paired comparison of two end-to-end arms. No arm is re-run."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.compare_refinement_arms import compare, load_arm, mcnemar_exact, paired_bootstrap


def arm(name: str, em_values: list[int], f1_values: list[float] | None = None) -> dict:
    f1_values = f1_values or [float(value) for value in em_values]
    return {
        "name": name,
        "config": {},
        "summary": {},
        "rows": {
            f"s{index}": {"sample_id": f"s{index}", "em_after": em, "f1_after": f1}
            for index, (em, f1) in enumerate(zip(em_values, f1_values))
        },
    }


class McNemarTest(unittest.TestCase):
    def test_no_disagreement_is_no_evidence(self):
        self.assertEqual(mcnemar_exact(0, 0), 1.0)

    def test_an_even_split_is_the_least_surprising_outcome(self):
        self.assertAlmostEqual(mcnemar_exact(10, 10), 1.0, places=6)

    def test_a_lopsided_split_is_significant(self):
        self.assertLess(mcnemar_exact(2, 30), 1e-5)

    def test_it_is_two_sided(self):
        self.assertAlmostEqual(mcnemar_exact(2, 30), mcnemar_exact(30, 2), places=12)

    def test_one_discordant_pair_proves_nothing(self):
        self.assertAlmostEqual(mcnemar_exact(0, 1), 1.0, places=12)

    def test_a_known_case_matches_the_binomial_by_hand(self):
        # 5 discordant pairs, all one way: 2 * (1/2)^5 = 0.0625
        self.assertAlmostEqual(mcnemar_exact(0, 5), 0.0625, places=12)


class PairedBootstrapTest(unittest.TestCase):
    def test_the_point_estimate_is_the_mean_paired_difference(self):
        result = paired_bootstrap([(0.0, 1.0), (0.0, 0.0), (1.0, 1.0), (0.0, 1.0)], iterations=200, seed=1)
        self.assertAlmostEqual(result["delta"], 0.5, places=4)

    def test_the_interval_brackets_the_estimate(self):
        result = paired_bootstrap([(0.0, 1.0)] * 50 + [(1.0, 0.0)] * 10, iterations=500, seed=1)
        self.assertLessEqual(result["ci95_low"], result["delta"])
        self.assertLessEqual(result["delta"], result["ci95_high"])

    def test_an_identical_arm_has_a_zero_width_interval(self):
        pairs = [(0.0, 0.0), (1.0, 1.0), (0.5, 0.5)]
        result = paired_bootstrap(pairs, iterations=200, seed=7)
        self.assertEqual((result["delta"], result["ci95_low"], result["ci95_high"]), (0.0, 0.0, 0.0))

    def test_it_reproduces_for_a_given_seed(self):
        pairs = [(0.0, 1.0), (1.0, 0.0), (0.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        self.assertEqual(paired_bootstrap(pairs, 300, seed=3), paired_bootstrap(pairs, 300, seed=3))

    def test_a_consistent_gain_excludes_zero(self):
        result = paired_bootstrap([(0.0, 1.0)] * 200, iterations=400, seed=5)
        self.assertGreater(result["ci95_low"], 0.0)


class CompareTest(unittest.TestCase):
    def test_it_counts_the_discordant_pairs_in_both_directions(self):
        baseline = arm("base", [1, 0, 1, 0, 1])
        other = arm("other", [0, 1, 1, 1, 1])
        result = compare(baseline, other, iterations=200, seed=42)
        self.assertEqual(result["only_baseline_correct"], 1)
        self.assertEqual(result["only_arm_correct"], 2)
        self.assertEqual(result["paired_samples"], 5)

    def test_it_pairs_on_sample_id_not_on_position(self):
        baseline = arm("base", [1, 0])
        other = arm("other", [0, 1])
        other["rows"] = {"s1": other["rows"]["s1"], "s0": other["rows"]["s0"]}
        result = compare(baseline, other, iterations=100, seed=42)
        self.assertEqual((result["only_baseline_correct"], result["only_arm_correct"]), (1, 1))

    def test_only_the_shared_samples_are_compared(self):
        baseline = arm("base", [1, 1, 1])
        other = arm("other", [0, 0])
        result = compare(baseline, other, iterations=100, seed=42)
        self.assertEqual(result["paired_samples"], 2)

    def test_disjoint_arms_are_an_error_rather_than_an_empty_verdict(self):
        baseline = arm("base", [1])
        other = arm("other", [1])
        other["rows"] = {"somewhere-else": {"sample_id": "somewhere-else", "em_after": 1, "f1_after": 1.0}}
        with self.assertRaises(ValueError):
            compare(baseline, other, iterations=10, seed=42)

    def test_the_reported_means_are_over_the_shared_samples(self):
        baseline = arm("base", [1, 0, 0, 0])
        other = arm("other", [1, 1, 0, 0])
        result = compare(baseline, other, iterations=100, seed=42)
        self.assertAlmostEqual(result["em_baseline"], 0.25, places=4)
        self.assertAlmostEqual(result["em_arm"], 0.5, places=4)


class LoadArmTest(unittest.TestCase):
    def test_a_missing_arm_lists_what_is_available(self):
        with self.assertRaises(FileNotFoundError) as caught:
            load_arm("no_such_arm_name")
        self.assertIn("Available", str(caught.exception))

    def test_an_explicit_path_is_read_as_given(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "arm.json"
            path.write_text(
                json.dumps({"config": {"keep_k": 5}, "summary": {"em_after": 0.2}, "rows": [{"sample_id": "a", "em_after": 1, "f1_after": 1.0}]}),
                encoding="utf-8",
            )
            loaded = load_arm(str(path))
        self.assertEqual(loaded["summary"]["em_after"], 0.2)
        self.assertEqual(list(loaded["rows"]), ["a"])


if __name__ == "__main__":
    unittest.main()
