"""README keeps the full Test table and reports the same scores without the 603 exposed ids."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from baselines._lib.answer_scoring import aggregate_answer_metrics
from baselines._lib.judge_io import summarize_results

ROOT = Path(__file__).resolve().parents[1]

SYSTEMS = (
    (
        "qwen_only",
        "baselines/outputs/qwen_only/predictions.jsonl",
        "baselines/outputs/judge/qwen_only_full/judge_results.jsonl",
    ),
    (
        "rag_matched",
        "baselines/outputs/rag_matched_test_context_fallback/predictions.jsonl",
        "baselines/outputs/judge/rag_matched_full/judge_results.jsonl",
    ),
    (
        "adaptive_product",
        "baselines/outputs/adaptive_product_test/predictions.jsonl",
        "baselines/outputs/judge/adaptive_product_full/judge_results.jsonl",
    ),
)


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


class ReadmeExposureTableTest(unittest.TestCase):
    def test_readme_exclusion_table_matches_current_files(self):
        split = json.loads(
            (ROOT / "baselines/splits/diagnosis_nq_train_74097_split.json").read_text(encoding="utf-8")
        )
        exposure = json.loads(
            (ROOT / "artifacts/controller/team_data_v1/known_exposure_ids.json").read_text(encoding="utf-8")
        )
        test_ids = [str(item) for item in split["splits"]["nq"]["test"]]
        exposed = set(test_ids) & {str(item) for item in exposure["observed_canonical_ids"]}
        self.assertEqual(len(test_ids), 11114)
        self.assertEqual(exposure["counts"]["test"], 603)
        self.assertEqual(len(exposed), 603)
        self.assertEqual(len(set(test_ids) - exposed), 10511)

        cells = []
        for mode, pred_rel, judge_rel in SYSTEMS:
            predictions = _load_jsonl(ROOT / pred_rel)
            judges = _load_jsonl(ROOT / judge_rel)
            self.assertEqual([str(row["sample_id"]) for row in predictions], test_ids)
            self.assertEqual([str(row["sample_id"]) for row in judges], test_ids)
            kept_predictions = [row for row in predictions if str(row["sample_id"]) not in exposed]
            kept_judges = [row for row in judges if str(row["sample_id"]) not in exposed]
            metrics = aggregate_answer_metrics(kept_predictions, mode=mode)
            judge = summarize_results(kept_judges)
            self.assertEqual(metrics["n"], 10511)
            aic = "—" if mode == "qwen_only" else _pct(metrics["answer_in_context_at_5"])
            cells.append(
                (
                    _pct(metrics["em"]),
                    _pct(metrics["f1"]),
                    _pct(metrics["normalized_accuracy"]),
                    aic,
                    _pct(judge["judge_assisted_accuracy"]),
                )
            )

        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("| Exact Match | 10.72% | 29.09% | 34.07% |", readme)
        self.assertIn("`n` is 11,114 in every file.", readme)
        expected_rows = (
            "| Exact Match | {0} | {1} | {2} |",
            "| Token F1 | {0} | {1} | {2} |",
            "| Normalized Accuracy | {0} | {1} | {2} |",
            "| AIC@5 | {0} | {1} | {2} |",
            "| Judge-assisted Accuracy | {0} | {1} | {2} |",
        )
        for template, values in zip(expected_rows, zip(*cells)):
            self.assertIn(template.format(*values), readme)


if __name__ == "__main__":
    unittest.main()
