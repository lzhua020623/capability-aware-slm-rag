# 重排分支 1d51856 兼容集成 · 2026-10-27

E 蒸馏准备首轮提交后，收尾刷新发现远端新增 `12aa2c6`、`56d952f`、`1d518562daa97ad555657a1965db4b8e02815c69` 三个提交。本轮合入当前 `controller_2`，保留上游原有历史和结果；不是新权重或新路径的效果晋升。

## 加入的内容

- 答案候选去重与 selector 双候选/多候选离线评测工具。
- 上游 selector 训练摘要和 2,367 条历史答案的评测摘要。
- `IndexCandidateProvider` 及 `--top-n` 可选实验：从原检索库补取更深候选，合并原 Top-5，再重排。默认仍只用原 Top-5。

历史摘要中 selector 多候选 EM 为 14.07%，单独重排组为 11.07%，但多候选使用多组已生成答案，成本和输入条件不同。双候选 selector 为 9.97%，低于重排组。权重及实际训练/调参 ID 没有随摘要交付，不能直接登记为新默认模型，也不能作为 E 蒸馏效果。

## 集成修正

1. 新 selector 评测先检查每个输入 arm 的全部 ID，再解码候选/参考；只能使用允许的 Train/Validation，不能通过 limit 隐藏 Test。所有 arm 必须具有同一 cohort、原答和参考。
2. 从源数据读题时先按所需 ID 过滤原始字节，未请求的 Validation、Test 不解码；参考与原始 source 对齐，模型只收到问题、证据和候选答案。
3. 新输出放在 team_nq_v1 目录，拒绝覆盖历史结果；记录实际 sample_ids、数据边界和权重来源尚未验真的状态。新本地默认模型路径是独立 team 候选，不改生产权重。
4. 深候选池的真实 loader 在导入原生/模型库前验证现有检索登记、全文件指纹、完整映射审计和路径，固定 BGE revision；读取后验证 FAISS 类型、维度及条数。独立虚构测试仍可注入轻量假 index。
5. `--top-n` 必须为正；现有 frozen 后端没有接入可变池，因此明确拒绝，不再把实际 Top-5 错报成 Top-N。

当前 Controller 没有启用 candidate_provider，额外检索也未接入生产计量器。此路径需另做真实资源/GPU、内存和计量验收；不能用独立脚本的耗时代替全系统 AC5。默认 BGE-large、Top-5、调用预算、P60 配置、E unavailable 均不变。

## 对 E 蒸馏的影响

500 条教师试产任务、500 条开发 ID、500 条选型 ID 和现有 v2a 契约不变。新增重排工具不能使用 E 的 Selection 清单做开发。旧 [E 准备归档](../artifacts/controller/e_distillation_v1/README.md)保留其测试提交和 1,281 项回归记录；本次更新的最终测试与来源指纹另存 [集成归档](../artifacts/controller/rerank_1d51856_v1/README.md)。

P60 两项原材料仍未齐；新 baseline 逐题输出仍待交付。因此最新源码的 CPU 集成检查不等于端到端生产可用，也不改变 E r2 的失败结论。
