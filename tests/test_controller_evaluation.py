"""Evaluate gains and damage without feeding references into runtime routing."""
from copy import deepcopy
import unittest

from scripts.evaluate_controller_integration import evaluate
from scripts.run_controller_integration import smoke_inputs
from tests.test_controller_integration import pipeline


class EvaluationTests(unittest.TestCase):
    def cases(self):
        nq, fever = smoke_inputs()
        records = [pipeline().run(nq, mode=mode) for mode in ("c1", "always_repair")]
        records += [pipeline().run(fever, mode="c1")]
        references = [{"dataset": "nq", "sample_id": nq.sample_id, "gold_answer_or_label": ["Amber"]},
                      {"dataset": "fever", "sample_id": fever.sample_id, "gold_answer_or_label": "REFUTES"}]
        return records, references

    def test_paired_gains_and_datasets_are_separate(self):
        records, refs = self.cases()
        report = evaluate(records, refs)["groups"]
        self.assertEqual(report["nq/c1"]["initial_correct"], 0)
        self.assertEqual(report["nq/always_repair"]["fixed"], 1)
        self.assertEqual(report["nq/always_repair"]["net_correct_gain"], 1)
        self.assertEqual(report["fever/c1"]["initial_correct"], 1)

    def test_regression_is_counted_as_damage(self):
        records, refs = self.cases()
        refs[0]["gold_answer_or_label"] = "Blue"
        report = evaluate(records, refs)["groups"]["nq/always_repair"]
        self.assertEqual((report["fixed"], report["regressed"], report["net_correct_gain"]), (0, 1, -1))

    def test_missing_or_duplicate_reference_and_prediction_rejected(self):
        records, refs = self.cases()
        for p, g in ((records, refs[:1]), (records, refs + refs[:1]), (records + records[:1], refs)):
            with self.assertRaises(ValueError):
                evaluate(p, g)

    def test_different_inputs_between_arms_are_rejected(self):
        records, refs = self.cases()
        for field in ("initial_answer", "input_sha256"):
            changed = deepcopy(records)
            changed[1][field] = "changed"
            with self.assertRaises(ValueError):
                evaluate(changed, refs)

    def test_incomplete_comparison_arm_is_rejected(self):
        records, refs = self.cases()
        extra = deepcopy(records[0])
        extra["sample_id"] = "extra"
        refs.append({**refs[0], "sample_id": "extra"})
        with self.assertRaises(ValueError):
            evaluate(records + [extra], refs)


if __name__ == "__main__":
    unittest.main()
