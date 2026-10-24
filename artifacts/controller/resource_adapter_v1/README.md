# Controller 资源接口验收 · 2026-10-24

本轮交付：NQ 全量 offset 定位校验、显式资源指纹注册、lazy Controller provider、开发查询真实检索重放；FEVER 原包导入、全量 ID/正文顺序核对及 FAISS 搜索回读验收。

`validation.json` 记录新增/改动源码及测试指纹、实际 CPU 环境、测试数量和各验证入口。`tests.log` 是完整回归，**706 项全部通过**；`targeted_tests.log` 是 NQ 注册/provider 与已有 Controller 接口的 31 项定向回归。

`protocol_check.json` 验证原 T1 的设计哈希及 68 个参考 Git blob；只报告 runner/providers 两处显式源码差异，不修改冻结设计。`data_check.json` 检查 T2 完整冻结及 552 个来源指纹。T2 旧报告中的资源缺口描述是冻结时快照，本轮通过追加验收记录更新进度，不重写历史数据。

本轮不修改原提示、分类器、阈值、预算、Top20/Top5 或样本划分。NQ 新增 `--nq-resources` 与旧 metadata 路径互斥，旧接口保留。新增读取层仅提供 passage_id/text；检索返回增加原分数，离线标注字段不进入运行时。

`local_mmap_search_diagnostic.log` / `local_heap_search_diagnostic.log` 为 CPU 重放耗时较长时的本机搜索诊断：两种加载方式均实际完成原全库搜索，未发生崩溃。这是并行负载下的一次诊断，不能作为受控速度比较；正式成本需在锁定的 GPU 执行环境中重新测量。

**仍未完成正式执行 seal**：FEVER provider、固定 FEVER 900 条新 C1、Full/No Probe/人工类型对照、严格全配对统计验收及 CUDA 环境锁定继续按 T1/T3 推进。CPU 与资源检查不构成准确率收益证明。
