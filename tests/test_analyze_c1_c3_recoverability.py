"""Tests for paired recoverability analysis and reusable cohorts."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.analyze_c1_c3_recoverability import analyze_dataset, pair_results


class RecoverabilityAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.c1 = self.root / "nq_c1.jsonl"
        self.c3 = self.root / "nq_c3.jsonl"
        c1_rows = []
        c3_rows = []
        for number in range(500):
            c1_correct = number >= 100
            c3_correct = number < 60 or number >= 100
            common = {
                "sample_id": str(number),
                "dataset": "natural_questions",
                "model_name": "Qwen/Qwen2.5-7B-Instruct",
                "question_or_claim": f"Question {number}?",
                "gold_answer_or_label": f"answer {number}",
            }
            c1_rows.append(
                dict(
                    common,
                    condition="C1",
                    evidence=[{"text": "retrieved"}],
                    prediction="c1",
                    correct=c1_correct,
                    em=int(c1_correct),
                    f1=float(c1_correct),
                )
            )
            c3_rows.append(
                dict(
                    common,
                    condition="C3",
                    evidence=[{"text": f"oracle {number}"}],
                    prediction="c3",
                    correct=c3_correct,
                    em=int(c3_correct),
                    f1=float(c3_correct),
                )
            )
        self.write(self.c1, c1_rows)
        self.write(self.c3, c3_rows)

    @staticmethod
    def write(path, rows):
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    @staticmethod
    def read_jsonl(path):
        return [json.loads(line) for line in path.read_text().splitlines()]

    def test_full_analysis_and_both_reusable_sets(self):
        summary = analyze_dataset(
            "nq",
            root=self.root,
            c1_path=self.c1,
            c3_path=self.c3,
            output_dir=self.root / "out",
        )
        self.assertEqual(summary["paired_samples"], 500)
        self.assertEqual(summary["total_c1_wrong"], 100)
        self.assertEqual(summary["c1_wrong_c3_correct"], 60)
        self.assertEqual(summary["c1_wrong_c3_wrong"], 40)
        self.assertEqual(summary["recoverable_failure_rate"], 0.6)
        recoverable = self.read_jsonl(self.root / "out/nq_c1_c3_recoverable.jsonl")
        persistent = self.read_jsonl(
            self.root / "out/nq_c1_c3_persistent_failures.jsonl"
        )
        self.assertEqual([row["sample_id"] for row in recoverable], [str(i) for i in range(60)])
        self.assertEqual(
            [row["sample_id"] for row in persistent], [str(i) for i in range(60, 100)]
        )
        self.assertTrue(all(row["oracle_evidence"] for row in persistent))

    def test_pairing_rejects_mismatched_or_duplicate_ids(self):
        rows = self.read_jsonl(self.c3)
        rows[-1]["sample_id"] = "missing-pair"
        self.write(self.c3, rows)
        with self.assertRaisesRegex(RuntimeError, "sample IDs differ"):
            pair_results(self.c1, self.c3, "natural_questions")

        rows[-1]["sample_id"] = rows[0]["sample_id"]
        self.write(self.c3, rows)
        with self.assertRaisesRegex(RuntimeError, "Duplicate"):
            pair_results(self.c1, self.c3, "natural_questions")


if __name__ == "__main__":
    unittest.main()
