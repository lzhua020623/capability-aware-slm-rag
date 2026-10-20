# E 最终改答审核清单

2026-10-21，最终75开发与一次性125挑战已完成冻结回放。全部主策略改答形成42条审核材料，涉及41个样本；代理辅助42/42；用户现已过目42条，正确性定案32条、待定10条。最新入口是 `F42_human_review.html`；原 `F42_review.html` 保留为人工确认前的冻结展示材料。

- F1–F14：开发75的 v7 改答，对照原 N 编号。
- F15：开发75的最终 r3 唯一改答，原 N96。
- F16–F42：挑战125的 v7 改答，保留 U 编号（冻结 challenge_number）。挑战 r3 没有改答。

F13/F15是同一题不同策略的不同答案，不重复算成两个独立样本。`prepare_answer_repair_final_review.py` 只会复用问题、初末答案、证据、参考和实际引用完全相同的审核材料；保留等价结果来源。本批没有完全相同的材料可合并。消融记录不在主审核范围，完整预测仍保存。

打开 `F42_review.html` 查看问题、C1、改答、代理意见、实际引用及完整Top-5，可按编号搜索。在聊天中回复“F1–F5同意”或逐条纠正即可。页面只读，不会自动提交人工标签。H前缀继续用于历史45条，N前缀只指原开发编号。

| 分层／策略 | 建议修复 | 疑似误修 | 正确保留 | 正确性待定 |
|---|---:|---:|---:|---:|
| 开发 v7 | 3 | 2 | 8 | 1 |
| 开发 r3 | 1 | 0 | 0 | 0 |
| 挑战 v7 | 2 | 4 | 15 | 6 |

以上是参考可见的代理建议，不能当作人工金标或最终收益。重点待定为F4、F19、F21、F22、F35、F36、F40；其他35条同样需要确认。F32还需确认FIFA项目及语料时点，F30需确认degree与diploma口径。

`pending_pairs.jsonl` 保存冻结材料、原结果SHA和全部pending人工字段。`assisted_partial.jsonl` 沿用历史工具命名，实际已有全部42条代理意见，含逐字来源定位。`manifest.json` 冻结并集范围；`assisted_summary.json` 记录后续代理完成状态。未知保留null，既有初始人工正确性不被代理覆盖。

重建材料使用 `scripts/prepare_answer_repair_final_review.py`；生成摘要及网页使用 `scripts/summarize_answer_repair_historical_review.py` 的 `--directory artifacts/answer_repair/final_review_v1 --prefix F --title 'E · 最终42条改答审核' --summary-version answer_repair_final_changed_assisted_complete_v1 --scope '<完整批次及限制说明>'` 参数。无新模型调用，无策略改动。

完整数字与限制见 `docs/ANSWER_REPAIR_RESULTS_20261021_FINAL.md`。人工确认、端到端路由实验及团队数据隔离协调仍待完成。

## 第一批人工确认更新

F12/F23/F25记录原答更好但改答正确性待定；F37按必须具体写Continental Army的受援对象口径，记两答都错。其余同意意见保留原有null。当前正确性未决为F4/F12/F19/F21/F22/F23/F25/F35/F36/F40。旧表为代理初稿快照，当前人工统计见 `data/annotations/human/answer_repair_final_v1_batch01_20261021.summary.json`，原话及澄清另存同名前缀 `.confirmations.json`，详细解释见 `docs/ANSWER_REPAIR_FINAL_HUMAN_REVIEW_20261021.md`。

开发v7确认修复3、误修2、待定2；开发r3确认修复1，无误修或待定；挑战v7确认修复2、误修3、待定8。未决项未按错误或正确补值，未计算跨策略汇总净收益。
