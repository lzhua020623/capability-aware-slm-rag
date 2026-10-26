# T5 选型结果 · 2026-10-26

首次私有 [Kaggle Version 1 / 362316572](https://www.kaggle.com/code/yearo111/controller-t5-selection-frozen-20261026?scriptVersionId=362316572) 已成功完成。页面记录运行 13,888.8 秒，约 3 小时 51 分钟；NQ、FEVER 各 200 道，五组共 2,000 条唯一记录，调用失败和预算超限为 0。本轮没有训练模型，也不是最终测试。

**唯一 NQ 0.2 路由候选不晋升，保留原路由。** 相对同一 E unavailable 模式的原 Adaptive，NQ 冻结 EM 为 93→90/200，修好 3、改坏 6，净 −3（−1.5 个百分点）；没有满足预登记的正净增条件。不改试 0.3／0.4，不因后续语义审核改写这次 EM 选型决定。

## 原始输出与核验

- [outputs.zip](outputs.zip)：原始下载字节，7,339,856 bytes，SHA256 `2139d926fb32c6071b95ceae3035ac95c06e7da10118b9c8976e302bcb62a7bd`。18 个 ZIP 成员，17 个文件校验全部一致。未覆盖原 T4、E r2 或运行包。
- 源码提交 `7c281d0193a4d1e205cfbc23dca31d6d4a0ac861`，输入包 SHA256 `151676a8c07008257086701c5dbced4fdc0b5e0f98f75f6e88caf98dcbae8f6f`。输入/执行封存、全部 400 题的 ID 与有效输入指纹、离线参考指纹逐一一致。
- 六个活动长输入路径通过，实际输入至少 16,005 tokens；两个停用 E 的零调用检查通过。两条历史查询的有序 Top-5 复现通过。云端预检使用原封存包，不含 QA gold。
- 400 条原 Adaptive 的进入状态、最终答案、最终证据和非 Probe 调用参数，与对应固定动作结果完全一致。候选复用这些结果和原有 Probe，无新 GPU 调用。
- 原 summary 从全部预测独立复算一致。不能把 `answer_replaced` 状态计数当作实际改答数，更不能当作成功修复数。

## 冻结评分

每个单元格是正确数／200。NQ 为 normalized EM，FEVER 为二分类标签准确率；两者分开报告。

| 比较组 | NQ | NQ 相对 C1 修好／改坏 | FEVER | FEVER 相对 C1 修好／改坏 |
|---|---:|---:|---:|---:|
| C1 缓存 | 89（44.5%） | 0／0 | 163（81.5%） | 0／0 |
| Always Retrieve | 89（44.5%） | 9／9 | 167（83.5%） | 7／3 |
| Always Rerank | 92（46.0%） | 7／4 | 164（82.0%） | 6／5 |
| Always Repair，E 停用 | 89（44.5%） | 0／0 | 163（81.5%） | 0／0 |
| 原 Adaptive | 93（46.5%） | 4／0 | 164（82.0%） | 1／0 |
| 新 NQ 0.2 候选，离线回放 | 90（45.0%） | 7／6 | 164（82.0%） | 1／0 |

原 Adaptive 相对 C1 的观察净增为 NQ +2.0、FEVER +0.5 个百分点；精确双侧 McNemar p 分别为 0.125、1.0。这是选型集描述，不能宣称已建立显著提升。候选相对原 Adaptive 的差值 95% 配对 bootstrap 区间为 [−4.5, +1.5] 个百分点，p=0.5078125；拒绝原因是未达到登记的选型门槛，并非证明统计显著变差。未运行 Full，不套用最终四项 Holm 家族，也不认证 AC5。

## 路由与成本

候选额外将 103 道 NQ 题送往补检索，实际改变 15 个答案。相比原 Adaptive，回放估算多 100 次 Qwen 调用、103 次检索、103 次评分，工具/模型调用耗时合计约增加 1,347 秒。原 Probe 保留；该时间是已有调用的回放估计，不是新系统部署时延。FEVER 逐题答案、动作和回放成本完全不变。

原 Adaptive 的实测增量调用为 NQ：79 次 Qwen、9 次检索、27 次评分；FEVER：111 次 Qwen、63 次检索、63 次评分。各组详细实测值见 [audit.json](audit.json)。所有成本均在缓存 C1 之后，不包含共同 C1 的历史生成成本。

## E 与语义审核边界

E 两数据集统一 unavailable，Always Repair 共 400 条保持输入答案，零 E 模型调用；原 Adaptive 的 10 条 NQ E 路由也回退保留，FEVER E 路由为 0。此结果不能证明 E 修复有效、核验准确或 AC6 达标。

[candidate_review15.jsonl](candidate_review15.jsonl) 保存候选相对原 Adaptive 的全部 15 个改答对、问题、原 Top-5、候选证据及冻结参考；人工语义字段全部 pending。NQ 的 6 条 EM 回退不等于已经确认 6 条真实语义误修，例如人名扩写与参考短名可能产生 EM 分歧。审核用于解释结果及报告，不将选型题转作下一轮训练或调参。T4 和 T9 的其他审核仍单独保留。

## 复现与后续

[audit.json](audit.json) 为完整自动核验与评分，[decision.json](decision.json) 为拒绝决定，[candidate_replay.jsonl](candidate_replay.jsonl) 为 400 条离线反事实结果；均不是新增 GPU 预测。新增 8 项回归在 `tests/test_controller_t5_output_audit.py`；完整 965 项测试全部通过（139.540 秒），原始 ZIP 在另一进程生成的五个结果文件逐字节一致。首次本地完整检查因缺少依赖失败的日志保留，补齐后 3 项失败路径复查及完整回归均通过；未重跑 GPU。验证记录见 [self_check.json](self_check.json)。

```bash
python scripts/audit_controller_t5_selection.py \
  --archive artifacts/controller/t5_selection_v1/kaggle_v1/outputs.zip \
  --output-dir /tmp/controller_t5_selection_independent_replay
```

该命令要求目标不存在以保护历史审计。下一步是保留原路由、收口审核、登记最终评测范围并进入 T7 封存；T8 尚未开始。蒸馏、RSI、E 语义修订均须另立扩展和数据隔离，不能继续用本次选型反馈搜索新候选。
