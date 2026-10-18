# v8：75 条 NQ Train 开发对照实验

这是真实修改后的 v8 开发包：原答案核验 → 最小修复 → 候选核验 → 不确定保留原答案。策略尚未晋升默认，GPU 结果及人工验收待完成。已将同哈希包上传为私有 Kaggle 数据。第 1 版在模型推理前因自动解压布局失败；加载修复后的第 2 版已在 T4 后台 Running，已确认 Qwen 开始生成，日志至少显示 v7 对照 15/75；完整四组结果仍待收取。当前状态见 `cloud_status.json`。

- 每组固定 75 条：11 个内部 E 目标、64 个正确对照；25 条待定排除。
- 四组：v7、v8 完整版、去原答案核验、去候选核验。C1 复用原始答案作为零调用基线。
- 预计 300 条输出，最多 600 次 Qwen 生成。调用预算不同，须报告实际成本。
- `answer_repair_v8_nq75_gpu.tar.gz`：冻结代码、输入、协议及离线初始审核分组。
- `manifest.json` / `preparation.json`：队列、策略、哈希、预期输出和调用预算。
- `validation.json`：从归档解压后进行的 96 项便携测试、四组 75 条 CPU 预检；不包含模型推理结果。
- `gpu_bundle/`：本机解压副本，按规则不入库。

配套 Notebook：`notebooks/answer_repair_v8_nq75.ipynb`。完整规则见 [v8 说明](../../../docs/ANSWER_REPAIR_V8.md)。输出汇总会生成 `comparison.json` 和 `output_review_queue.jsonl`，新答案的人工结论保留 pending。

包 SHA-256：`906f096f78e36a4f272cdd863ce2e7383d62acb638c1e480a06c5d3f75daa7a3`。

这是开发实验；旧 C1 人工标签不能给新答案背书，75 条的收益也不能当作未见评测结论。该目录的 preparation 与 manifest 是运行前快照，云端执行状态另存，不能修改为“已验收”。

## Kaggle 加载修复 r1

第 1 版取得 T4 后运行 30.5 秒，在模型推理之前因归档查找失败退出，0 条预测。Kaggle 已自动把 tar.gz 解压为目录。`kaggle_v1_failure.json` 保留失败记录。

当前运行入口改为 `notebooks/answer_repair_v8_nq75_transport_r1.ipynb`，可加载完整归档或自动解压的目录；后者用固定的 checksums 文档哈希校验全部 85 个有效负载文件，并只复制这些文件到可写目录。原包、75 条输入、模型、提示及全部实验参数不变。`transport_r1.json` 固定新 Notebook 与加载器的哈希；`transport_validation.json` 记录原包真实解压布局的加载验证。
