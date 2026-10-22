# v9 独立受约束输出接口实验

这是12个全新虚构样例的工程实验，不是NQ/FEVER正式评测，不是模型训练，不回用退役smoke10/8。原v9三份快照和v7/r3/r4包均保持冻结；新入口不会晋升默认策略。

复用v9完整Top-5、分解、隔离核验及严格后检查。两种策略分别使用prompt-only与XGrammar 0.2.8解码，**两组提示均含相同JSON Schema文本**，避免把提示差异当作解码收益。模型/revision/NF4/seed/384 token预算不变，最多144次调用、48条最终预测。此处没有直接重答组，不与旧smoke数字拼接。

生成约束包括decision枚举、合法证据ID、数组长度、FEVER修复标签及keep时原答常量；重复ID/问题、空理由、不变repair和NQ原文定位仍由旧后检查拒绝。JSON合法和来源合法不能证明语义支持。

**GPU前门槛**：48条齐全、冻结回放通过、预算不超限，24条受约束结果至少22条通过后检查，Unknown ID与keep改写为0。失败即存档，不再围绕本批调提示；通过只允许另行冻结真实语义开发，不能据此声称准确率提高。

本地已用固定Qwen tokenizer（152064词表）完成36份原生语法编译/正例接受，24个错误keep/非法ID反例拒绝；详见 [CPU核验](cpu_grammar_check.json)。没有加载Qwen模型权重，**不是CUDA/NF4兼容性或真实生成通过**。26项Python检查、前端校验通过；工程样例不会混入200条真实标注质量审核。

GPU包及notebook已冻结。Kaggle [私有任务](https://www.kaggle.com/code/yearo111/e-contract-v1-synthetic-20261022) Version 1已提交，观察到T4×2、联网、Queued。实际状态以 [cloud_status.json](cloud_status.json) 及收取回执为准；当前尚无收齐并回放的GPU结果。

文件：`synthetic_cases.jsonl`、`preflight.json`、`cpu_grammar_check.json`、`bundle_manifest.json`、`answer_repair_contract_v1_gpu.tar.gz`。notebook在 `notebooks/answer_repair_contract_v1.ipynb`。包记录40个源文件哈希；原始模型生成、环境与导出结果收齐后单独归档。

```sh
python scripts/run_answer_repair_contract.py --check-only --out artifacts/answer_repair/v9_contract_v1/preflight.json
python scripts/run_answer_repair_contract.py --compile-only --out /tmp/contract_cpu.json
# 以下需要CUDA、项目固定依赖和xgrammar==0.2.8
python scripts/run_answer_repair_contract.py --out results/answer_repair/contract_v1/predictions.jsonl
python scripts/run_answer_repair_contract.py --replay --out results/answer_repair/contract_v1/predictions.jsonl
```

采用 [XGrammar官方HF接入方式](https://xgrammar.mlc.ai/docs/latest/start/quick_start.html)。新代码与冻结包见 `src/compensation/constrained_evidence_id.py`、`qwen_constrained.py` 和 `scripts/run_answer_repair_contract.py`。原有接口语义不足仍需真实数据核验，本地语法通过不解除旧v9的停止规则。
