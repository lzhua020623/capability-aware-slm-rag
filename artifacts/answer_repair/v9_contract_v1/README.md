# v9 独立受约束输出接口实验

这是12个全新虚构样例的工程实验，不是NQ/FEVER正式评测，不是模型训练，不回用退役smoke10/8。原v9三份快照和v7/r3/r4包均保持冻结；新入口不会晋升默认策略。

复用v9完整Top-5、分解、隔离核验及严格后检查。两种策略分别使用prompt-only与XGrammar 0.2.8解码，**两组提示均含相同JSON Schema文本**，避免把提示差异当作解码收益。模型/revision/NF4/seed/384 token预算不变，最多144次调用、48条最终预测。此处没有直接重答组，不与旧smoke数字拼接。

生成约束包括decision枚举、合法证据ID、数组长度、FEVER修复标签及keep时原答常量；重复ID/问题、空理由、不变repair和NQ原文定位仍由旧后检查拒绝。JSON合法和来源合法不能证明语义支持。

**GPU前门槛**：48条齐全、冻结回放通过、预算不超限，24条受约束结果至少22条通过后检查，Unknown ID与keep改写为0。失败即存档，不再围绕本批调提示；通过只允许另行冻结真实语义开发，不能据此声称准确率提高。

本地已用固定Qwen tokenizer（152064词表）完成36份原生语法编译/正例接受，24个错误keep/非法ID反例拒绝；详见 [CPU核验](cpu_grammar_check.json)。没有加载Qwen模型权重，**不是CUDA/NF4兼容性或真实生成通过**。26项Python检查、前端校验通过；工程样例不会混入200条真实标注质量审核。

GPU包及原notebook保持冻结。Kaggle [私有任务](https://www.kaggle.com/code/yearo111/e-contract-v1-synthetic-20261022) Version 1（361060970）运行573.2秒后显示failure；已下载全部71个原始输出和完整日志。**48条预测及云端回放已完成，失败只发生于最后导出ZIP。** 原tar将源码mtime固定为0（1970年），notebook中的ZipFile默认拒绝早于1980年的文件，因此内部导出只留下22字节空ZIP。使用Kaggle的Download output收取完整工作目录，结果没有丢失。

本地Python3.13冻结回放48/48通过，40个源码文件及校验清单与输入包逐字节一致，汇总重算一致。实际96次调用，记录的output_tokens_estimate最大109，受约束调用含停止token最大110，均在144次调用／384 token预算内。原始ZIP SHA-256为 `864f9f3ca01b305be3434accb06c4b01ee55bdad320f5b88b88fa6aec687b1bd`。

| 策略／解码 | 严格后检查合规 | 调用数 | 累计生成秒数 |
|---|---:|---:|---:|
| evidence_id／prompt_only | 10/12 | 12 | 62.50 |
| evidence_id／XGrammar | 11/12 | 12 | 66.19 |
| factored_evidence_id／prompt_only | 10/12 | 36 | 126.93 |
| factored_evidence_id／XGrammar | 11/12 | 36 | 130.58 |

**受约束组22/24合规，Unknown ID与keep改写均为0，达到本批预声明工程门槛。** prompt_only为20/24；新增合规项是同一S02在两个策略下不再改写keep答案，不能视为两个独立问题的收益。剩余两项受约束拒绝同属S03：模型给出数字9，选中原文写nine，仍被原有逐字片段规则拒绝。合规结果包含保留、缺证和冲突；合规率不是答题准确率。分解组本次36次调用，不能称为与12次调用的单轮组算力匹配。

本地回放不是新生成；本批12个虚构题没有真实benchmark收益或人审结论。只允许接下来另行冻结真实语义开发协议，不默认晋升策略，不回用本批调提示/放宽原文规则，不解除旧smoke10的退役要求。接下来继续标签质量审核，再在可信开发标签上检查错答误放、正确答案误拒、修复／误修及成本。

原始文件在 [kaggle_20261022](kaggle_20261022/)，[完整接收与回放回执](result_validation.json)、[重算汇总](release_audit/recomputed_summary.json)、[本地验证代码](release_audit/verify_export.py)。完整输出ZIP为 `kaggle_outputs_v1_361060970.zip`；原空内层ZIP仍包含在其中。收取过程没有改写原输出、gold、人工标签或模型。

已新增独立 `notebooks/answer_repair_contract_v1_exportfix.ipynb`，仅将ZipFile设为 `strict_timestamps=False`；原notebook、GPU tar及推理代码未改动。回归测试用实际notebook的finally代码重现1970年时间戳报错，再验证修复后的ZIP保留同一路径和内容、跳过pycache。详见 [导出修复回执](bundle_manifest_exportfix.json)。该变体未提交GPU运行：现有48条已经完整，不需要重跑。

文件：`synthetic_cases.jsonl`、`preflight.json`、`cpu_grammar_check.json`、`bundle_manifest.json`、`answer_repair_contract_v1_gpu.tar.gz`。notebook在 `notebooks/answer_repair_contract_v1.ipynb`。包记录40个源文件哈希；原始模型生成、环境与导出结果收齐后单独归档。

```sh
python scripts/run_answer_repair_contract.py --check-only --out artifacts/answer_repair/v9_contract_v1/preflight.json
python scripts/run_answer_repair_contract.py --compile-only --out /tmp/contract_cpu.json
# 以下需要CUDA、项目固定依赖和xgrammar==0.2.8
python scripts/run_answer_repair_contract.py --out results/answer_repair/contract_v1/predictions.jsonl
python scripts/run_answer_repair_contract.py --replay --out results/answer_repair/contract_v1/predictions.jsonl
# 用与云端相同的Python major/minor收取并离线回放：此处为3.13，无需GPU。
python3.13 artifacts/answer_repair/v9_contract_v1/release_audit/verify_export.py
# 只生成导出修复变体，保留原notebook；本批无需重跑模型。
python scripts/build_answer_repair_contract_notebook.py --export-fix
```

采用 [XGrammar官方HF接入方式](https://xgrammar.mlc.ai/docs/latest/start/quick_start.html)。新代码与冻结包见 `src/compensation/constrained_evidence_id.py`、`qwen_constrained.py` 和 `scripts/run_answer_repair_contract.py`。原有接口语义不足仍需真实数据核验，本地语法通过不解除旧v9的停止规则。
