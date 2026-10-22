# v9 紧凑上下文工程快照

仅压缩首版重复的来源元数据，保留完整 Top-5、句子 ID、原文偏移映射及语义规则。相同五条 NQ＋五条 FEVER、三组各五条；不按 EM 或样本语义错误调提示。

- 输入：`answer_repair_v9_dev_v2_gpu.tar.gz`，SHA-256 `682f0911a24006271d6b72f46c01e5768fe10bd0f5c1cd9730067e539c5a3a68`。
- 自包含 notebook：`answer_repair_v9_smoke_compact.ipynb`，嵌入并校验上述输入包，保持私有。由 `scripts/build_answer_repair_v9_notebook.py` 生成。
- `implementation_validation.json`：13 项核心／共 78 项相关检查，六组预检，与首版函数和输入字节比较。
- `cloud_status.json`：实际提交状态，Kaggle Version 2／`361004665`。

Version 2 于 495.4 秒正常结束，`kaggle_20261022_v2/outputs.zip` SHA-256 为 `3e483f8544bc2cb8ecd2ae8f0988db95635dcdaabf6e4851287558f3b0e52207`。30 条齐全、六组冻结本地回放通过；但 20 条结构化输出中 19 条不合规、全部保留 C1，不能评价为修复成功。原始输出、日志、逐文件指纹与分类成本汇总已保留。

压缩格式中同时展示段落键 `R1` 和句子键 `R1S1` 引入来源命名歧义，另立 v3 接口候选，不回改本包结果。完整开发搁置；GPU 完成不表示有净收益。详见 [工程报告](../../../docs/ANSWER_REPAIR_V9_ENGINEERING_20261022.md)。
