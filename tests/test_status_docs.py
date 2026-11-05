"""The status docs name the formal outputs that are already in this directory."""
from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class StatusDocSentencesTest(unittest.TestCase):
    def test_project_status_splits_baseline_results_from_unmerged_rerank(self):
        text = (ROOT / "docs/PROJECT_STATUS_AND_DIRECTION.md").read_text(encoding="utf-8")
        self.assertIn(
            "其中 baseline `10601c5` 的 11,114 条预测、指标和 Judge 已在本目录 `baselines/outputs/`。"
            "重排 `d4d9e45` 比已合入的 `1d51856` 仍多 11 个提交，这一项尚未合并。",
            text,
        )
        self.assertIn(
            "SFT未开始。正式 11,114 条测试中的 generation_repair 已运行，1575 条状态为 success。"
            "原C1、路由、团队数据隔离不变。",
            text,
        )
        self.assertNotIn("这两项尚未合并", text)
        self.assertNotIn("E生产模式仍unavailable", text)

    def test_team_data_boundaries_record_delivered_baseline_outputs(self):
        text = (ROOT / "docs/TEAM_DATA_BOUNDARIES.md").read_text(encoding="utf-8")
        self.assertIn(
            "最终配对比较使用的 predictions、metrics、run_config 已在 `baselines/outputs/`："
            "Qwen-only、配套 RAG、自适应系统各 11,114 条，Judge 在 `baselines/outputs/judge/`。",
            text,
        )
        self.assertIn(
            "FEVER 的团队划分、P60 的原始在线特征和模型使用 ID、候选重排/选择器的训练来源仍是后续待办。"
            "新 baseline 的 11,114 条逐题结果和 Judge 已在 `baselines/outputs/`。"
            "正式测试中的 generation_repair 已执行；本轮体检没有启用新的训练权重。",
            text,
        )
        self.assertNotIn("当前缺的是新 baseline 逐题输出", text)
        self.assertNotIn("新 baseline 逐题交付仍是后续待办", text)
        self.assertNotIn("E 仍 unavailable", text)


if __name__ == "__main__":
    unittest.main()
