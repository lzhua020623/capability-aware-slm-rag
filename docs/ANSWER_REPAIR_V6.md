# E v6：一次抽取与独立 MNLI 判定

本轮是 v5 负结果后的固定开发实验。保留旧代码策略与冻结实验包，新增两个显式可选的运行策略。人工结果验收按用户要求暂缓，自动检查不代替人工结论。v6r1 仅补充 JSON 环境字段的断点续跑兼容、成本元数据及回放工具；模型、提示词、阈值、名单和策略均不改。原 v6 归档保留，但未确认真实 GPU 执行。

## 预先固定的策略

- `extractive_reanswer`，仅 NQ：同一 Qwen2.5-7B-Instruct / NF4 / FP16 / greedy / seed 42，最多一次 384-token 生成。只输出 `spans`；程序把通过校验的原文字符片段作为答案，多个片段用 ` ; ` 连接。最多 3 个片段，每个最多 80 词，总共最多 120 词。保留词语、否定和标点，允许空格与大小写的格式差异。拒绝不存在、重叠、重复和超长片段。引用可追溯不等于语义正确，选择了错误事件仍可能通过。
- `fever_nli`，仅 FEVER：独立 `microsoft/deberta-large-mnli`，revision `7296194b9009373def4f7c5dad292651e4b5cf4e`，FP32、eval、batch 4、阈值固定 0.90。claim 始终作为完整 hypothesis，每段 C1 evidence 是 premise，超长证据以 64 token 重叠分窗，保留所有 token，不截断 claim。任一窗口 entailment ≥ 0.90 且无 contradiction ≥ 0.90，输出 SUPPORTS；反向则 REFUTES；双向高分冲突或均无高分则保留初始答案。0.90 是未经 FEVER 校准的分类器分数阈值，不代表 90% 事实正确率；跨窗口、多跳推断和同名实体仍可能失败。窗口原文、位置、三类概率、模型版本与成本全部保存。

NLI 的官方模型卡注明 MNLI fine-tuning；选它是为了避免显式 FEVER 监督模型引入额外任务标签。官方资料不能证明预训练语料与本项目完全无重叠，不能宣称绝对无污染。

来源：[固定版本模型卡](https://huggingface.co/microsoft/deberta-large-mnli/blob/7296194b9009373def4f7c5dad292651e4b5cf4e/README.md)、[官方实现](https://github.com/microsoft/DeBERTa)、[标签映射](https://huggingface.co/microsoft/deberta-large-mnli/blob/7296194b9009373def4f7c5dad292651e4b5cf4e/config.json)。

NLI 是增加独立模型的实验，不是相同 SLM 预算的公平替代对照。`extra_calls/latency/input_tokens` 仅统计 Qwen；`verifier_calls/verifier_latency/verifier_input_tokens` 单独统计分类器 forward batch、批处理墙钟延迟（含分批编码、传输与前向，不含预先分窗）和未 padding 输入 token。模型下载/初始化时间不在逐条前向延迟中，完整运行日志保留。`--max-new-tokens 1` 是 NLI CLI 占位参数，无生成、不产生 token。

## 固定数据及停止条件

1. 重用 v5 的 28 条已审核训练开发样本：FEVER 14，NQ 14。来源、人工纳入理由及旧 EM 不一致项不变。
2. 额外跑冻结旧 dev50：FEVER 50、NQ 50，观察正确样本误修。与第 1 批重叠 FEVER 1、NQ 4；不能把 128 条输出当成独立样本，也不能把重复样本的修复算两次。
3. 本轮只运行这两个预先固定的策略；不根据结果反复修改阈值或追求小开发集的最高分。不触碰 Diagnosis validation/test。
4. 自动结果归档、CPU 确定性回放、Controller 边界检查和交付文档完成后，工程工作可交接；人工语义验收与团队最终 held-out 验收单列 pending。不得用这一轮开发结果宣称总体修复有效。

## 代码及复现

- `src/compensation/extractive_repair.py`：NQ 提示词和字符片段回填。
- `src/compensation/nli_repair.py`：NLI 分窗、GPU 推断、冲突回退与离线回放。
- `src/compensation/answer_repair.py`：统一入口；`controller_adapter.py` 接受相同策略，仍只执行上游 `generation_repair`。
- `scripts/run_answer_repair.py`：预检、带版本指纹的断点续跑、原始输出和成本。

```sh
python3.11 -m unittest discover -s tests
python3.11 scripts/prepare_reviewed_answer_repair.py --experiment v6 --manifest configs/splits/answer_repair_reviewed_v6.json --out-dir artifacts/answer_repair/v6r1_reviewed --bundle-name answer_repair_v6r1_reviewed_gpu
python3.11 scripts/build_reviewed_answer_repair_notebook.py --experiment v6 --bundle artifacts/answer_repair/v6r1_reviewed/answer_repair_v6r1_reviewed_gpu.tar.gz --out notebooks/answer_repair_colab_v6r1_reviewed.ipynb
```

运行状态与结论见 `ANSWER_REPAIR_RESULTS_20261014.md`；归档包中的本文是运行前协议，不随之后结果重写。
