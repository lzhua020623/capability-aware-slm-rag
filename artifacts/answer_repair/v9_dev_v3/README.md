# v9 单一来源命名空间接口候选

针对 v2 的来源标识歧义，将提示改为按文档标题分组，只展示可引用句子 ID；补明精确 ID、最多五个事实引用与 decision 小写。完整 Top-5、标题、偏移追溯、语义规则、解析器和回退机制、模型、样本及预算保留。

- 冻结输入：`answer_repair_v9_dev_v3_gpu.tar.gz`，SHA-256 `4dab82e46dd6b48b0a880c5de38de36418d0c9189c027f12c63ebdfe53e4a940`。
- notebook：`answer_repair_v9_smoke_contract.ipynb`，自包含私有实验包。
- `implementation_validation.json`：78 项检查、六组冻结预检，证据原文、标题及句子 ID 不变。
- `engineering_gate.json`：提交 GPU 前声明的门槛和停止规则；不按 EM 或题目语义表现调提示。

复现旧版需使用其包内源码。候选未晋升默认策略；完整开发和语义收益尚未确认。详见 [工程报告](../../../docs/ANSWER_REPAIR_V9_ENGINEERING_20261022.md)。

Version 3（361008203）已在 523.2 秒正常结束，30／30 条输出、六组本地冻结回放通过。结构化输出仅 11／20 合规，低于预声明 18／20 门槛；来源 ID 不匹配错误归零。已按停止规则冻结，不继续调这十题，完整开发未启动。

结果位于 `kaggle_20261022_v3/`：`outputs.zip` SHA-256 为 `9b1a4a117fd018ca89b9bc6f12db0fa50f47c13591c2e3207d8346287bb2572a`；原始结果／日志、逐文件指纹、本地回放及工程汇总均已保存。三条改答对应两道 NQ，辅助审核已写入 `changed_assisted_review.jsonl` 和 [审核网页](kaggle_20261022_v3/changed_review.html)，尚不计入人工确定修复数。

`collection_receipt.json` 绑定输入包、预声明门槛、原始输出和派生审核文件指纹。回放使用 CPython 3.13.16（GPU 为 3.13.15）：从包内目录运行 `scripts/run_answer_repair_v9.py --replay` 并指定原 `--dataset/--strategy/--all/--c1/--out`。回放重用保存的模型文本，验证处理及评分可复算，不是重新生成，也不保证其他软件版本下 GPU 输出逐字一致；重新推理须另存运行快照。
