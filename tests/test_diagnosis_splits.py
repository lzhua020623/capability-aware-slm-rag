"""Tests for the deterministic Diagnosis v1 split. Uses synthetic records only."""

from __future__ import annotations

import unittest
from collections import Counter

from scripts import build_diagnosis_splits as splits


def record(sample_id: str, query: str, label: str = "correct", review: bool = False) -> dict:
    return {
        "input": {"sample_id": sample_id, "dataset": "fever", "query": query, "evidence": [], "initial_answer": "SUPPORTS"},
        "target_label": label,
        "metadata": {"needs_manual_review": review},
    }


def synthetic_dataset() -> list[dict]:
    rows = [record(f"c{i}", f"Claim about subject{i} in region{i}.") for i in range(60)]
    rows += [record(f"u{i}", f"Unrelated statement on topic{i} near place{i}.", "evidence_utilisation") for i in range(10)]
    rows += [
        record("dup1", "Giada at Home aired on a TV channel."),
        record("dup2", "Giada at Home aired on a TV channel."),
        record("near1", "Ryan Gosling has been to Chad.", "evidence_insufficiency"),
        record("near2", "Ryan Gosling has been to country Chad.", "correct"),
        record("flagged", "Flagged sample for review.", "evidence_accessibility", review=True),
    ]
    return rows


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.data = {"fever": synthetic_dataset()}
        self.split = splits.build_split(self.data)

    def test_validation_passes(self):
        splits.validate_split(self.split, self.data)

    def test_each_trusted_sample_exactly_once_and_flagged_excluded(self):
        ids = [sample_id for name in splits.SPLITS for sample_id in self.split["splits"]["fever"][name]]
        self.assertEqual(len(ids), len(set(ids)))
        trusted = {r["input"]["sample_id"] for r in splits.trusted_records(self.data["fever"])}
        self.assertEqual(set(ids), trusted)
        self.assertNotIn("flagged", ids)

    def test_duplicate_and_near_duplicate_groups_stay_together(self):
        where = {sid: name for name in splits.SPLITS for sid in self.split["splits"]["fever"][name]}
        self.assertEqual(where["dup1"], where["dup2"])
        self.assertEqual(where["near1"], where["near2"])

    def test_query_groups(self):
        groups = splits.query_groups(splits.trusted_records(self.data["fever"]))
        multi = sorted(sorted(r["input"]["sample_id"] for r in g) for g in groups if len(g) > 1)
        self.assertEqual(multi, [["dup1", "dup2"], ["near1", "near2"]])

    def test_deterministic(self):
        self.assertEqual(self.split, splits.build_split(self.data))
        self.assertNotEqual(self.split["splits"], splits.build_split(self.data, seed=7)["splits"])

    def test_ratio_and_stratification(self):
        counts = Counter(name for name in splits.SPLITS for _ in self.split["splits"]["fever"][name])
        total = sum(counts.values())
        self.assertAlmostEqual(counts["train"] / total, 0.70, delta=0.03)
        where = {sid: name for name in splits.SPLITS for sid in self.split["splits"]["fever"][name]}
        utilisation = Counter(where[f"u{i}"] for i in range(10))
        self.assertEqual(utilisation, Counter({"train": 7, "validation": 2, "test": 1}))

    def test_split_targets_sum(self):
        for size in (1, 2, 17, 49, 338, 402):
            self.assertEqual(sum(splits.split_targets(size).values()), size)
        self.assertEqual(splits.split_targets(17), {"train": 12, "validation": 3, "test": 2})

    def test_leakage_detected(self):
        broken = {**self.split, "splits": {"fever": {k: list(v) for k, v in self.split["splits"]["fever"].items()}}}
        broken["splits"]["fever"]["test"].append(broken["splits"]["fever"]["train"][0])
        with self.assertRaises(ValueError):
            splits.validate_split(broken, self.data)

    def test_split_file_has_ids_only(self):
        self.assertEqual(set(self.split), {"version", "seed", "ratios", "filter", "grouping", "splits"})
        for ids in self.split["splits"]["fever"].values():
            self.assertTrue(all(isinstance(sample_id, str) for sample_id in ids))


if __name__ == "__main__":
    unittest.main()
