"""Tests for the accessibility rerank training-set builder. No model is loaded.

The two properties that matter here are supervision correctness (the positive
really is the answer-bearing passage) and leak control (nothing close to a
frozen Dev/Test query is trained on, and no near-duplicate question straddles
the split).
"""

from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from scripts import build_rerank_training_data as builder


def passage(rank: int, text: str, passage_id: str | None = None) -> dict:
    return {"rank": rank, "passage_id": passage_id or f"p{rank}", "text": text, "score": 1.0 - 0.1 * rank}


def record(
    sample_id: str = "s1",
    label: str = "evidence_accessibility",
    gold_answer: str = "William Shakespeare",
    gold_passage_id: str | None = None,
    evidence: list[dict] | None = None,
    query: str = "who wrote hamlet",
) -> dict:
    return {
        "sample_id": sample_id,
        "dataset": "nq",
        "query": query,
        "initial_answer": "Christopher Marlowe",
        "primary_label": label,
        "evidence": evidence
        or [
            passage(1, "Christopher Marlowe was an English playwright."),
            passage(2, "The Globe Theatre staged many plays."),
            passage(3, "Hamlet is a tragedy written by William Shakespeare."),
        ],
        "annotation_reference": {"gold_answer": gold_answer, "gold_passage_id": gold_passage_id},
    }


class ContainsAnswerTest(unittest.TestCase):
    def test_ignores_case_punctuation_and_articles(self):
        self.assertTrue(builder.contains_answer("william shakespeare", "by William Shakespeare ."))
        self.assertTrue(builder.contains_answer("The Globe", "the globe theatre"))

    def test_requires_a_contiguous_run(self):
        self.assertFalse(builder.contains_answer("Shakespeare Hamlet", "Hamlet by William Shakespeare"))
        self.assertFalse(builder.contains_answer("Ben Jonson", "Hamlet by William Shakespeare"))

    def test_empty_answer_never_matches(self):
        self.assertFalse(builder.contains_answer("", "anything"))
        self.assertFalse(builder.contains_answer("   ", "anything"))


class JaccardAndGroupingTest(unittest.TestCase):
    def test_jaccard(self):
        self.assertEqual(builder.jaccard(builder.tokens("a b"), builder.tokens("a b")), 1.0)
        self.assertEqual(builder.jaccard(builder.tokens("a b"), builder.tokens("c d")), 0.0)
        self.assertEqual(builder.jaccard(frozenset(), builder.tokens("a")), 0.0)

    def test_near_duplicate_queries_share_a_group(self):
        queries = [
            "who wrote the novel moby dick",
            "who wrote the novel moby dick ?",
            "when was the eiffel tower built",
        ]
        groups = builder.group_queries([builder.tokens(query) for query in queries], 0.6)
        self.assertEqual(groups[0], groups[1])
        self.assertNotEqual(groups[0], groups[2])

    def test_unrelated_queries_stay_separate(self):
        queries = ["who sang purple rain", "what is the capital of peru", "how deep is lake baikal"]
        groups = builder.group_queries([builder.tokens(query) for query in queries], 0.6)
        self.assertEqual(len(set(groups)), 3)


class BuildExamplesTest(unittest.TestCase):
    def test_positive_is_the_answer_bearing_passage(self):
        examples, reasons = builder.build_examples([record()], "evidence_accessibility", 4)
        self.assertEqual(len(examples), 1)
        example = examples[0]
        self.assertEqual(example["positive"]["rank"], 3)
        self.assertEqual(example["positive_source"], "gold_answer_text")
        self.assertEqual([negative["rank"] for negative in example["negatives"]], [1, 2])
        self.assertEqual(reasons["positive_rank_3"], 1)

    def test_falls_back_to_the_gold_passage_id(self):
        examples, reasons = builder.build_examples(
            [record(gold_answer="an answer stated nowhere", gold_passage_id="p2")], "evidence_accessibility", 4
        )
        self.assertEqual(examples[0]["positive"]["passage_id"], "p2")
        self.assertEqual(examples[0]["positive_source"], "gold_passage_id")
        self.assertEqual(reasons["positive_from_gold_passage_id"], 1)

    def test_other_labels_are_ignored(self):
        rows = [record(label=label) for label in ("correct", "evidence_insufficiency", "evidence_utilisation")]
        examples, reasons = builder.build_examples(rows, "evidence_accessibility", 4)
        self.assertEqual(examples, [])
        self.assertEqual(reasons["candidates"], 0)

    def test_samples_without_a_positive_are_skipped(self):
        examples, reasons = builder.build_examples(
            [record(gold_answer="nowhere in any passage", gold_passage_id=None)], "evidence_accessibility", 4
        )
        self.assertEqual(examples, [])
        self.assertEqual(reasons["skipped_no_positive"], 1)

    def test_samples_without_a_negative_are_skipped(self):
        single = record(evidence=[passage(1, "Hamlet was written by William Shakespeare.")])
        examples, reasons = builder.build_examples([single], "evidence_accessibility", 4)
        self.assertEqual(examples, [])
        self.assertEqual(reasons["skipped_no_negative"], 1)

    def test_the_gold_passage_is_never_used_as_a_negative(self):
        rows = [record(gold_answer="William Shakespeare", gold_passage_id="p2")]
        examples, _ = builder.build_examples(rows, "evidence_accessibility", 4)
        self.assertNotIn("p2", [negative["passage_id"] for negative in examples[0]["negatives"]])

    def test_max_negatives_keeps_the_hardest_distractors(self):
        examples, _ = builder.build_examples([record()], "evidence_accessibility", 1)
        self.assertEqual([negative["rank"] for negative in examples[0]["negatives"]], [1])
        unlimited, _ = builder.build_examples([record()], "evidence_accessibility", 0)
        self.assertEqual([negative["rank"] for negative in unlimited[0]["negatives"]], [1, 2])

    def test_duplicate_answer_bearing_passages_are_all_excluded_from_negatives(self):
        rows = [
            record(
                evidence=[
                    passage(1, "William Shakespeare wrote it."),
                    passage(2, "A distractor."),
                    passage(3, "Hamlet is by William Shakespeare."),
                ]
            )
        ]
        examples, _ = builder.build_examples(rows, "evidence_accessibility", 4)
        self.assertEqual(examples[0]["positive"]["rank"], 1)  # best = highest-ranked positive
        self.assertEqual([negative["rank"] for negative in examples[0]["negatives"]], [2])


class SplitTest(unittest.TestCase):
    def test_groups_never_straddle_splits(self):
        examples = [record(sample_id=str(index)) for index in range(100)]
        group_ids = [index // 2 for index in range(100)]  # pairs share a group
        splits = builder.split_examples(examples, group_ids)
        seen: dict[int, str] = {}
        for name, rows in splits.items():
            for row in rows:
                group = int(row["sample_id"]) // 2
                self.assertEqual(seen.setdefault(group, name), name)
        self.assertEqual(sum(len(rows) for rows in splits.values()), 100)

    def test_ratios_are_roughly_seventy_fifteen_fifteen(self):
        examples = [record(sample_id=str(index)) for index in range(1000)]
        splits = builder.split_examples(examples, list(range(1000)))
        self.assertEqual(len(splits["train"]), 700)
        self.assertEqual(len(splits["validation"]), 150)
        self.assertEqual(len(splits["test"]), 150)

    def test_split_is_deterministic(self):
        examples = [record(sample_id=str(index)) for index in range(50)]
        group_ids = list(range(50))
        first = builder.split_examples(examples, group_ids)
        second = builder.split_examples(examples, group_ids)
        self.assertEqual({k: [r["sample_id"] for r in v] for k, v in first.items()},
                         {k: [r["sample_id"] for r in v] for k, v in second.items()})


class HeldOutFilterTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "held_out.jsonl"
        rows = [{"sample_id": "d1", "question": "who wrote the novel moby dick"}]
        self.path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
        self.addCleanup(self.directory.cleanup)

    def test_exact_and_near_duplicate_queries_are_excluded(self):
        too_close, size = builder.held_out_filter(self.path, 0.6)
        self.assertEqual(size, 1)
        self.assertTrue(too_close(builder.tokens("who wrote the novel moby dick")))
        self.assertTrue(too_close(builder.tokens("Who wrote the novel Moby Dick?")))
        self.assertTrue(too_close(builder.tokens("who wrote the novel moby dick book")))

    def test_unrelated_queries_pass(self):
        too_close, _ = builder.held_out_filter(self.path, 0.6)
        self.assertFalse(too_close(builder.tokens("when was the eiffel tower built")))

    def test_reads_the_query_field_too(self):
        path = Path(self.directory.name) / "query_field.jsonl"
        path.write_text(json.dumps({"sample_id": "d2", "query": "how tall is mount everest"}), encoding="utf-8")
        too_close, _ = builder.held_out_filter(path, 0.6)
        self.assertTrue(too_close(builder.tokens("how tall is mount everest")))


class RareTokenBlockTest(unittest.TestCase):
    def test_items_are_indexed_under_their_rarest_tokens(self):
        token_sets = [builder.tokens("the quick zebra"), builder.tokens("the quick fox"), builder.tokens("the quick fox")]
        blocks = builder.rare_token_blocks(token_sets, per_item=1)
        self.assertEqual(blocks["zebra"], [0])
        self.assertEqual(sorted(blocks["fox"]), [1, 2])
        self.assertNotIn("the", blocks)

    def test_union_find_is_transitive(self):
        groups = builder.UnionFind(3)
        groups.union(0, 1)
        groups.union(1, 2)
        self.assertEqual(len({groups.find(index) for index in range(3)}), 1)


class CounterContractTest(unittest.TestCase):
    def test_reasons_counter_reports_every_candidate(self):
        rows = [record(sample_id="a"), record(sample_id="b", gold_answer="missing"), record(sample_id="c", label="correct")]
        examples, reasons = builder.build_examples(rows, "evidence_accessibility", 4)
        self.assertIsInstance(reasons, Counter)
        self.assertEqual(reasons["candidates"], 2)
        self.assertEqual(len(examples), 1)


if __name__ == "__main__":
    unittest.main()
