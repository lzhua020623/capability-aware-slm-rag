"""The status docs name the formal outputs that are already in this directory."""
from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class StatusDocSentencesTest(unittest.TestCase):
    def test_project_status_splits_baseline_results_from_unmerged_rerank(self):
        text = (ROOT / "docs/PROJECT_STATUS_AND_DIRECTION.md").read_text(encoding="utf-8")
        self.assertIn(
            "the 11,114 predictions, metrics and Judge of baseline `10601c5` are already in `baselines/outputs/` in this directory. "
            "Reranking `d4d9e45` is still 11 commits ahead of the merged `1d51856`; this item has not yet been merged.",
            text,
        )
        self.assertIn(
            "SFT has not started. In the formal 11,114-item test, generation_repair has been run, and 1575 items have status success. "
            "The original C1, routing and team data isolation are unchanged.",
            text,
        )
        self.assertNotIn("这两项尚未合并", text)
        self.assertNotIn("E生产模式仍unavailable", text)

    def test_team_data_boundaries_record_delivered_baseline_outputs(self):
        text = (ROOT / "docs/TEAM_DATA_BOUNDARIES.md").read_text(encoding="utf-8")
        self.assertIn(
            "the predictions, metrics and run_config used for the final paired comparison are already in `baselines/outputs/`: "
            "Qwen-only, the companion RAG and the adaptive system with 11,114 items each, with Judge in `baselines/outputs/judge/`.",
            text,
        )
        self.assertIn(
            "Team splitting for FEVER, P60's raw online features and model usage IDs, and the training provenance of candidate reranking/selectors are still later to-dos. "
            "The 11,114 per-question results and Judge of the new baseline are already in `baselines/outputs/`. "
            "generation_repair in the formal test has been executed; this round's health check did not enable new training weights.",
            text,
        )
        self.assertNotIn("当前缺的是新 baseline 逐题输出", text)
        self.assertNotIn("新 baseline 逐题交付仍是后续待办", text)
        self.assertNotIn("E 仍 unavailable", text)


if __name__ == "__main__":
    unittest.main()
