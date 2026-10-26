"""Tests for the answer selector's data construction and scoring. No model is loaded."""

from __future__ import annotations

import unittest

from scripts.build_selector_training_data import build_examples, gold_answer
from scripts.train_answer_selector import ANSWER_MARKER, claim_text, evidence_text, summarise_scores


def record(sample_id="s1", initial="Christopher Marlowe", gold="William Shakespeare", label="evidence_accessibility"):
    return {
        "sample_id": sample_id,
        "query": "who wrote hamlet",
        "initial_answer": initial,
        "primary_label": label,
        "evidence": [{"text": "Hamlet is a tragedy by William Shakespeare ."}, {"text": "A distractor ."}],
        "annotation_reference": {"gold_answer": gold},
    }


def by_source(examples):
    return {item["source"]: item for item in examples}


class GoldAnswerTest(unittest.TestCase):
    def test_it_reads_the_offline_reference(self):
        self.assertEqual(gold_answer(record()), "William Shakespeare")

    def test_a_missing_reference_is_empty(self):
        self.assertEqual(gold_answer({"annotation_reference": None}), "")
        self.assertEqual(gold_answer({}), "")


class BuildExamplesTest(unittest.TestCase):
    def test_a_wrong_stored_answer_yields_gold_positive_and_two_negatives(self):
        examples, reasons = build_examples([record(), record(sample_id="s2", gold="Ben Jonson")])
        mine = by_source([item for item in examples if item["sample_id"] == "s1"])
        self.assertEqual(mine["gold_answer"]["label"], 1)
        self.assertEqual(mine["gold_answer"]["answer"], "William Shakespeare")
        self.assertEqual(mine["model_answer_wrong"]["label"], 0)
        self.assertEqual(mine["model_answer_wrong"]["answer"], "Christopher Marlowe")
        self.assertEqual(mine["borrowed_gold"]["label"], 0)
        self.assertEqual(reasons["records"], 2)

    def test_a_correct_stored_answer_becomes_the_positive(self):
        examples, _ = build_examples([record(initial="William Shakespeare"), record(sample_id="s2", gold="Ben Jonson")])
        mine = by_source([item for item in examples if item["sample_id"] == "s1"])
        self.assertIn("model_answer_correct", mine)
        self.assertEqual(mine["model_answer_correct"]["label"], 1)
        self.assertNotIn("gold_answer", mine)
        self.assertNotIn("model_answer_wrong", mine)  # the model answer is the positive, not a negative

    def test_the_borrowed_negative_is_never_right_for_this_sample(self):
        # Every other sample's gold equals this sample's gold, so no borrowed negative is possible.
        records = [record(sample_id=f"s{index}") for index in range(5)]
        examples, reasons = build_examples(records)
        self.assertEqual(reasons["skipped_no_borrowed_negative"], 5)
        self.assertFalse([item for item in examples if item["source"] == "borrowed_gold"])

    def test_records_without_a_gold_answer_are_skipped(self):
        examples, reasons = build_examples([record(gold="")])
        self.assertEqual(examples, [])
        self.assertEqual(reasons["skipped_no_gold"], 1)

    def test_every_sample_carries_both_classes(self):
        records = [record(sample_id="s1"), record(sample_id="s2", gold="Ben Jonson")]
        examples, _ = build_examples(records)
        for sample_id in ("s1", "s2"):
            labels = {item["label"] for item in examples if item["sample_id"] == sample_id}
            self.assertEqual(labels, {0, 1}, sample_id)

    def test_gold_style_strings_appear_on_both_sides(self):
        records = [record(sample_id="s1"), record(sample_id="s2", gold="Ben Jonson")]
        examples, _ = build_examples(records)
        golds = {"William Shakespeare", "Ben Jonson"}
        self.assertTrue(any(item["answer"] in golds and item["label"] == 1 for item in examples))
        self.assertTrue(any(item["answer"] in golds and item["label"] == 0 for item in examples))

    def test_the_build_is_deterministic(self):
        records = [record(sample_id=f"s{index}", gold=f"gold {index}") for index in range(10)]
        first, _ = build_examples(records)
        second, _ = build_examples(records)
        self.assertEqual(first, second)

    def test_examples_keep_the_runtime_fields(self):
        examples, _ = build_examples([record(), record(sample_id="s2", gold="Ben Jonson")])
        item = examples[0]
        self.assertEqual(item["query"], "who wrote hamlet")
        self.assertEqual(item["evidence"], ["Hamlet is a tragedy by William Shakespeare .", "A distractor ."])


class InputFormatTest(unittest.TestCase):
    def test_the_claim_holds_the_question_and_the_answer(self):
        row = {"query": "who wrote hamlet", "answer": "William Shakespeare"}
        self.assertEqual(claim_text(row), "who wrote hamlet" + ANSWER_MARKER + "William Shakespeare")

    def test_evidence_is_joined_in_order(self):
        self.assertEqual(evidence_text({"evidence": ["a", "b"]}), "a b")


class SummariseScoresTest(unittest.TestCase):
    def test_pick_at_1_counts_the_best_scoring_answer_per_sample(self):
        rows = [
            {"sample_id": "a", "label": 1},
            {"sample_id": "a", "label": 0},
            {"sample_id": "b", "label": 1},
            {"sample_id": "b", "label": 0},
        ]
        self.assertEqual(summarise_scores(rows, [2.0, 1.0, 0.5, 1.5])["pick_at_1"], 0.5)

    def test_samples_without_a_choice_are_excluded(self):
        rows = [{"sample_id": "a", "label": 1}, {"sample_id": "b", "label": 0}, {"sample_id": "b", "label": 0}]
        stats = summarise_scores(rows, [1.0, 1.0, 2.0])
        self.assertEqual(stats["samples_with_a_choice"], 0)
        self.assertIsNone(stats["pick_at_1"])

    def test_accuracy_thresholds_at_zero(self):
        rows = [{"sample_id": "a", "label": 1}, {"sample_id": "a", "label": 0}]
        self.assertEqual(summarise_scores(rows, [1.0, -1.0])["accuracy"], 1.0)
        self.assertEqual(summarise_scores(rows, [-1.0, 1.0])["accuracy"], 0.0)

    def test_a_length_mismatch_is_an_error(self):
        with self.assertRaises(ValueError):
            summarise_scores([{"sample_id": "a", "label": 1}], [1.0, 2.0])

    def test_empty_input_does_not_divide_by_zero(self):
        stats = summarise_scores([], [])
        self.assertIsNone(stats["pick_at_1"])
        self.assertIsNone(stats["accuracy"])


if __name__ == "__main__":
    unittest.main()
