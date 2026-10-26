# T5 用户采用辅助审核 · 2026-10-26

当前入口：[已确认的 17 题网页](review/review17.html)。用户在查看 R01–R17 辅助判断后回复“按照你判断的来”，本快照仅落实这次明确采用，原始辅助意见和未审快照原样保留。

- 已采用辅助意见：17/17；这是可见参考、有代理辅助的审核，不是独立盲审。
- 完整定案：11/17。R01、R02、R04、R05、R08、R12、R13、R14 的原答和补检索答案都正确；R07、R10、R15 为原答错误、补检索答案正确。
- 尚未完整定案：R03、R06、R09、R11、R16、R17。R06、R09 的原答和 Adaptive 判错，补检索答案仍待定；另外四题的三项对错均待定。R16、R17 维持“倾向保留原答”的操作建议，不把这种偏好写成正确性标签。
- 确认范围仅有 C1、原 Adaptive、Always Retrieve 的语义判断。网页中的 Always Rerank 和历史 Always Repair 仍未在本次确认范围内。

这是特意选出的重点队列。11 道定案题里的 3 道修复、0 道误修，不能代表全部 200 道开发题、107 次候选补检索或路由候选的净收益。六题待定不按正确计分；所有冻结 gold、别名和 EM 原样保留，也不自动转成训练/DPO 标签。

## 可追溯记录

- [用户确认回执](confirmation.json)：原话、范围、原辅助文件及备注 SHA256。
- [入库逐题记录](review/review17.jsonl)：`assistance_adopted_by_user` 与 `confirmed / partial / unresolved` 分开记录。
- [统计](review/summary.json) 与 [输出指纹](review/manifest.json)。
- [原始辅助快照](../t5_assisted_review_v1/README.md)仍保留，不把历史“未获用户确认”改写为当时已经确认。

本轮只保存审核，不修改模型、运行时路由、r2 验收、E 的 unavailable 决定或选型规则。候选尚未冻结／晋升，之后需结合待定项及审核覆盖范围做回放分析；不能把接受辅助意见等同于无条件允许送选型。

重建命令（目标目录必须不存在）：

```bash
python scripts/record_controller_t5_confirmation.py --confirmation artifacts/controller/t5_human_review_v1/confirmation.json --output-dir /tmp/controller-t5-confirmed-new
```
