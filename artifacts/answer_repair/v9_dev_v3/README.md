# v9 单一来源命名空间接口候选

针对 v2 的来源标识歧义，将提示改为按文档标题分组，只展示可引用句子 ID；补明精确 ID、最多五个事实引用与 decision 小写。完整 Top-5、标题、偏移追溯、语义规则、解析器和回退机制、模型、样本及预算保留。

- 冻结输入：`answer_repair_v9_dev_v3_gpu.tar.gz`，SHA-256 `4dab82e46dd6b48b0a880c5de38de36418d0c9189c027f12c63ebdfe53e4a940`。
- notebook：`answer_repair_v9_smoke_contract.ipynb`，自包含私有实验包。
- `implementation_validation.json`：78 项检查、六组冻结预检，证据原文、标题及句子 ID 不变。
- `engineering_gate.json`：提交 GPU 前声明的门槛和停止规则；不按 EM 或题目语义表现调提示。

复现旧版需使用其包内源码。候选未晋升默认策略；完整开发和语义收益尚未确认。详见 [工程报告](../../../docs/ANSWER_REPAIR_V9_ENGINEERING_20261022.md)。
