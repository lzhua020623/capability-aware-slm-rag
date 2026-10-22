import unittest
from scripts.summarize_answer_repair_v9_judgments import summarize


class JudgmentDiagnosticsTests(unittest.TestCase):
    def test_uncertainty_cannot_masquerade_as_successful_classification(self):
        rows = [{"sample_id": "1", "dataset": "nq", "extra_calls": 3, "changed": False,
                 "status": "insufficient_evidence", "verdict": "insufficient"}]
        result = summarize(rows, [{"sample_id": "1", "initial_correct": False}], "nq")
        self.assertEqual(result["classified_coverage_of_human_scored"], 0)
        self.assertEqual(result["counts"]["wrong_original_retained"], 1)

    def test_unscored_labels_are_excluded_and_change_is_not_called_regression(self):
        rows = [{"sample_id": str(n), "dataset": "fever", "extra_calls": 1,
                 "changed": True, "status": "repaired", "verdict": "repair"} for n in (1, 2)]
        labels = [{"sample_id": "1", "initial_correct": True}, {"sample_id": "2", "initial_correct": None}]
        result = summarize(rows, labels, "fever")
        self.assertEqual(result["counts"]["human_initial_scored"], 1)
        self.assertEqual(result["counts"]["without_human_initial_label"], 1)
        self.assertEqual(result["counts"]["false_reject_correct"], 1)
        self.assertNotIn("regressions", result["counts"])

    def test_duplicate_result_ids_are_rejected(self):
        row = {"sample_id": "1", "dataset": "nq", "extra_calls": 1,
               "changed": False, "status": "kept", "verdict": "keep"}
        with self.assertRaises(ValueError):
            summarize([row, row], [{"sample_id": "1", "initial_correct": True}], "nq")


if __name__ == "__main__":
    unittest.main()
