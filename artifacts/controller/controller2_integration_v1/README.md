# Controller 2 集成验证 · 2026-10-27

来源与范围见 [交接](../../../docs/CONTROLLER2_INTEGRATION.md) 和 `merge_sources.json`。

- **完整回归：1,224 项全部通过，104.634 秒**，`full_tests.log`。
- 补检索专项 50 项、重排与集成专项 329 项、历史封存专项 10 项通过；专项之间有重叠，不累计为独立样本或测试数。
- 6 项新增合并边界回归覆盖类型迁移、拒绝候选留痕、同 ID 文本裁剪、FEVER 支持、E unavailable 与默认模型配置。
- 旧 T5 包装回归改为恢复原始封存源码；新增检查确认新 Controller 源码会被旧封存拒绝。`full_tests_initial_frozen_guard.log` 保留初次被严格哈希拦截的回归记录。
- 历史 Windows 包测试来源设置、缺失 NLTK/pandas 的修正前日志也保留，没有放宽生产源码哈希校验。
- `before.json` 为开工状态，`self_check.json` 为收工检查，`source_snapshot.json` 固定实际测试的源码树；`checksums.json` 核对本目录文件。
- `environment.json` 记录 CPU 测试环境、NLTK 3.9.2 及英文 punkt_tab 的四个资源文件 SHA-256；NLTK 数据和模型缓存不入库。

两条来源分支均进入新分支历史。`controller` 原分支、旧冻结配置、包、历史结果、AGENTS.md 及未提交用户审核文件均保留。来源报告字节与上游一致；未启用新重排权重/裁剪/答案选择器，未生成教师数据或启动训练/GPU/API 调用。

生产预检查仍按预期退出 2：P60 原始在线特征代码与三个模型的训练/调参 ID 未交付；E 对两个数据集仍为 unavailable。`readiness_after.json` 不代表 GPU 工程或效果验收通过。

下一步是团队划分及逐题 baseline 接入、跨路线训练 ID 隔离、P60 材料补齐，以及 E 教师资格验证。当前集成不作为准确率提升声明。
