# 质量第 1 轮：确认集结果（2026-10-30）

冻结配置 `configs/controller/quality_baseline_v1.json`（commit 67b1c9e）在登记的确认集 2,440 题（Validation）上只跑了一次，结果通过登记的规则：判官口径净值大于 0。

| | C1 | 产品最终 | 变化 |
|---|---:|---:|---:|
| 判官口径正确 | 1647 (67.5%) | 1692 (69.3%) | **+45** |
| 修好 / 改坏 | | | 102 / 57（精确 McNemar p = 0.00045） |

- 送去增强 851 题，全部走 P3-20（检索 20 + 重排 + Top-5 + 重新作答）；其中 421 题答案改变。
- 扫描集 6,000 题上同一配置净 +99（+1.65 点），确认集净 +45（+1.84 点），量级一致。
- 判官：团队 `run_llm_judge.py`（`origin/feature/baseline-evaluation`，未改动），gpt-6.1-sol。自动匹配或判 CORRECT 算对，UNCERTAIN 不算对。2,861 个（题目，答案）组合，1,766 次 API 调用；两批同时运行时有 97 条遇到 429 限流，用同一命令补齐，最终无失败。
- 运行：RTX 5090，2026-10-29T14:29Z 到 2026-10-29T20:48Z，正式运行、未续跑。金答案只在本地读取，没有发到 GPU 服务器。

## 文件

- `results.tar.gz`：GPU 原始结果和清单（`run/`）、团队格式预测和指标（`scored/`）、判官输入和输出（`judge/`）。
- `score.json`：判官口径汇总和判定；`provenance.json`：登记、ID、结果和判官的哈希。
- 复算：`scripts/score_quality_eval.py score --predictions scored/predictions.jsonl --judged judge/outputs/shard_*/judge_results.jsonl`；测试 `tests/test_quality_confirm_release.py` 会逐项回放。

下一步按登记：团队 Test 11,114 题只跑一次，报告为暴露后重评（Test 曾被逐题分析过）。
