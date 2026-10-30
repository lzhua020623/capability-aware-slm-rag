# BM25 新证据替换：Test 结果（2026-10-31）

登记 `configs/controller/bm25_gate_v1.json`（commit 63b0169，在任何 Test GPU 运行之前提交）。规则：路线 1 送去 P3-20 的题中，只有当 L2 `bge20_bm2520_ce_only` 的 Top-5 含有 BM25 带来的段落（不在 BGE 前 20 内）时，才改用 L2 的答案；其余题保持路线 1 的答案。

**按登记的采纳标准通过**：与路线 1 在 Test 上配对，判官口径净 **+105**（修好 226 / 改坏 121），精确 McNemar p = 1.9e-08。

| | 判官口径正确 | 本规则相对它的净值（修好 / 改坏） |
|---|---:|---:|
| 路线 1（B） | 6632 (59.7%) | +105（226 / 121） |
| 团队 RAG 基线 | 6391 | +346（656 / 310），+3.11 点，p = 3.1e-29 |
| 团队现有产品 | 6445 | +283（595 / 312），+2.55 点，p = 3.7e-21 |
| 本规则 | **6737 (60.6%)** | |

- 送去 P3 的 6,166 题中，1,639 题的 Top-5 含 BM25 段落并重新作答，764 题答案与路线 1 不同。
- 同一规则的回放：扫描集 +39（93 / 54），确认集 +16（43 / 27），按题量换算与 Test 量级一致。

## 披露

- 这条规则是在 L2 整体未通过确认集（`artifacts/controller/l2_confirm_v1`）之后，按机制拆分得到的，确认集不是独立证据。Test 此前已被逐题分析过，本次是路线 1 Test 之后又一次披露的使用；路线 2 的部分 Test 输出仍未评分。
- 生成脚本加了 `--data-role test`，只有显式指定时才解锁封存的 Test 边界，与路线 1 Test 运行器的写法相同（commit 6d2e966、2dcf483）。两次启动在边界检查处停止，没有产生输出。
- 标签顺序与路线 1 Test 相同，团队 RAG 标签最后（以 RAG 为准）。本批 577 个（题目，答案）组合，API 452 次，无失败。
- 金答案只在本地使用；GPU 服务器只收到题号、问题和段落 ID。

## 文件

- `results.tar.gz`：`selections.jsonl`（每题 L2 Top-5 和其中 BM25 段落）、`candidates/manifest.json`、`generations/`、`judge/`、`report.json`、`replay/`（扫描集和确认集回放）。
- 原始候选（约 60 MB）留在本地 `data/processed/quality_sweep_v1/bm25_test/`，哈希记在 `provenance.json`。
- 测试 `tests/test_bm25_gate_test_release.py` 用路线 1 Test 归档和本归档逐项回放评分和三组比较。
