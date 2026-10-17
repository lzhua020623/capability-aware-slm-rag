# E 的 Verification 与 Answer Repair

E 对应 Evidence Utilisation：模型已经看到可用证据，初始回答却可能取错实体、误解关系、忽视否定或输出不受证据支持的答案。Diagnosis 由其他成员负责，Controller 决定何时调用 E。

截至 2026-10-16，工作基础为 v7，下一轮 v8 尚未实现。v7 主包与第 4 批已完成一次 GPU 运行，541+300 条结果通过各自冻结代码回放；两种策略均未晋升默认，人工语义验收仍待完成。当前入口为 [交接清单](ANSWER_REPAIR_HANDOFF.md)、[v7 结果](ANSWER_REPAIR_RESULTS_20261014_V7.md) 和 [统一审核总表](../artifacts/answer_repair/v7_combined_review/review_all.txt)。

已确认的 37 条早期候选中，19 条核心错误候选与 9 条正确对照已用于开发，9 条暂缓。v6r1 关键案例的用户确认见 [确认清单](ANSWER_REPAIR_CONFIRMATION_20261014.md)。这些已有开发依据可用于工程改进；新语义规则仍需审核关键争议案例，并另行验收。

下文解释共用接口与早期单次 JSON 核验行为；`short_answer_quote` 和 `fever_nli_anchored` 的具体行为以 [v7 协议](ANSWER_REPAIR_V7.md) 为准，独立模型成本单列。旧版设计、失败分析和执行记录见 [历史归档](history/answer_repair/README.md)，不能把旧的“待运行”状态当成当前进度。

## 输入输出及处理规则

`repair_answer(DiagnosisInput, generator)` 直接复用 [共享接口](DIAGNOSIS_PROBE_INTERFACE.md)。输入只含 sample ID、dataset、query、原始证据及 initial answer。适配器显式选择字段，不把 gold、C0/C3、correctness、retrieval gold hit 或大模型结果传给模型。

证据内容和顺序保持 C1 原样。模型输出：

```json
{"verdict":"repair","answer":"Paris","citations":[{"rank":1,"quote":"The capital of France is Paris."}]}
```

| Verdict | 首版行为 |
|---|---|
| `keep` | 引用支持原答案的证据，原答案必须保持不变 |
| `repair` | 提供修复答案和引用；NQ 的新答案必须是引用中的完整文字片段，FEVER 必须恰好为 SUPPORTS 或 REFUTES |
| `insufficient` | 不强制改变答案，记录证据不足状态，由上层处理 unresolved 或后续操作 |

引用的 rank 必须存在，quote 必须出现在对应 passage 中；匹配允许大小写和空白差异，保留词语和标点。格式错误、伪造引用、非法标签或无依据的新 NQ span 均回退到初始答案并记录 `invalid_output`。Backend 异常直接停止，可续跑；不会伪装成模型选择 keep。没有可用证据时零调用。

这个例子只解释接口，不是实验结果。真实 NQ C1 中也有类似任务理解错误，例如问题询问 “what is the sentencing reform act of 1984”，初始回答却给出 “U.S. Sentencing Commission”，而现有证据同时描述该法律和它设立的委员会。E 应核对问题所问对象和证据中关系，再选择回答片段，不能仅检查两个实体是否都出现。

引用检查证明片段来自输入，不能证明片段蕴含答案。FEVER 尤其需要核对实体、关系、否定、数量和多证据组合。首版日志固定标记 `independent_support_status=not_checked`，不会把引用合法当作 NLI entailment、faithfulness 或运行时答案正确的证明。v6r1 的显式独立 MNLI 分支已实测，但 dev50 出现两条错误实体导致的误修；高分仍不是人工真值，支持性和校准尚需后续验收。

## 与 Diagnosis 和 Controller 的边界

最终集成使用 `src/compensation/controller_adapter.py` 的 `generation_repair(request, decision, generator)`。`decision` 是上游完成 Diagnosis、Probe 和 Safe Binary 后的 `ControllerDecision`；E 仅执行 action=`generation_repair`，其他动作返回原答案且零调用。上游 confidence 即使低于旧阈值，E 也不重复判断；路由政策由 Controller 负责。调用方应复用同一个已加载 generator。

离线 runner 用 `--controller-decisions path.jsonl` 接收 `ControllerDecision.to_dict()` 的七个字段，文件应覆盖本次评测所有 ID。未知字段、缺失决策或 ID 不匹配均报错，支持可选 dataset 字段。示例：

```json
{"sample_id":"1","action":"generation_repair","diagnosis_label":"evidence_utilisation","diagnosis_confidence":0.2,"used_probe":true,"probe_overrode_diagnosis":false,"needs_compensation":true}
```

在整合上游 Controller/Probe 分支后，可只加载一次模型，给 Probe 注入相同模型的原始 greedy 调用，给 E 使用带 JSON 停止器的 generator。Probe 的 prompt、160-token 预算、trigger 和 Safe Binary 不变：

```python
from src.compensation.qwen_repair import QwenRepairGenerator
from src.compensation.controller_adapter import generation_repair
from src.controller.controller import run_controller
from src.probe.probe import EvidenceProbe, PROBE_MAX_NEW_TOKENS
from src.generation.qwen import generate_greedy

e_generator = QwenRepairGenerator()
probe = EvidenceProbe(lambda prompt: generate_greedy(
    e_generator.tokenizer, e_generator.model, prompt,
    max_new_tokens=PROBE_MAX_NEW_TOKENS,
)[0])
decision = run_controller(item, classifier, probe)
e_result = generation_repair(item, decision, e_generator)
```

这是整合用法，当前 E 分支没有复制上游 Controller/Probe 实现。四种 action 与真实上游 `ControllerDecision` 类已在隔离目录完成确定性 fixture 验证；另已完成真实 GPU Controller→E 开发试跑，27 次 Probe 和 1 次 E，无净收益，见 [独立实验报告](history/answer_repair/ANSWER_REPAIR_RESULTS_20261013.md)。外部 model loader、classifier 和 input 对象均由调用方提供。其他两类补偿没有在该试跑中执行，不能称为完整 adaptive 系统评测。

旧 `--diagnoses` 仅用于独立开发消融，不能替代最终 Controller 路由。它接收如下 runtime output JSONL，dataset 使用 `nq`/`fever`，并覆盖所有本次评测 ID：

```json
{"sample_id":"1","dataset":"nq","diagnosis_label":"evidence_utilisation","confidence":0.85,"needs_probe":false}
```

仅主标签为 utilisation 且 `needs_probe=false` 时执行 E；其他标签保留初始答案并零调用，等待 Probe 的样本也跳过。阈值和 Probe 策略由 Controller 管理，E 不自行设置 confidence threshold。缺失 diagnosis 会报错。

`--all` 是明确的 Always-Verify 开发消融，不要求 diagnosis 分类器存在，也不读取 gold 来挑选错误样本。它可先验证 E 的效果，然后再接团队真实的 diagnosis 输出。人工标签用于离线 oracle-routing 消融时，需要独立命名和报告，不能当作 runtime diagnosis。

## 文件

| 文件 | 用途 |
|---|---|
| `src/compensation/answer_repair.py` | 纯 Python 策略入口、核验规则与回退 |
| `src/compensation/qwen_repair.py` | 延迟加载共享 Qwen NF4 generator，记录 token 和延迟 |
| `src/compensation/controller_adapter.py` | 接收最终 ControllerDecision，仅执行 E 动作 |
| `scripts/run_answer_repair.py` | 读 C1、可选路由、预检、落盘、锁和断点续跑 |
| `src/evaluation/answer_repair.py` | 独立的离线配对评分与汇总 |
| `tests/test_answer_repair.py` | 行为、信息隔离、路由和中断恢复测试 |

代码要求 Python 3.11+，与 diagnosis 的 `StrEnum` 接口一致。预检和测试只需标准库，不加载 torch、transformers、FAISS、模型或数据集。

## 当前电脑可直接执行

在仓库根目录：

```bash
python3.11 scripts/run_answer_repair.py --dataset nq --source-ref origin/feature/-ce-70B --all --check-only
python3.11 scripts/run_answer_repair.py --dataset fever --source-ref origin/feature/-ce-70B --all --check-only
python3.11 -m unittest discover -s tests -v
```

`--source-ref` 直接从该提交中读取 archived C1，包括已记录的 Top-5 全文。无需重新检索或下载 Wikipedia/index。亦可用 `--c1 path/to/c1.jsonl` 读取本地 C1 交接文件。`--dry-run` 输出第一条 prompt，仍不调用模型、不保存预测，避免把模拟输出混入真实结果。

## GPU 机器执行

### 首轮便携实验包（2026-10-13）

本机为 Apple Silicon Mac，未安装 PyTorch 或 Transformers，没有 CUDA；真实修复在 Colab 执行。准备脚本会冻结开发样本，并打包当前代码和已有 C1 全文。下面为首版建立命令；已存在的版本包拒绝被不同代码覆盖：

```bash
python3.11 scripts/prepare_answer_repair_experiment.py
```

修订版复用原始开发 ID，另建包：

```bash
python3.11 scripts/prepare_answer_repair_experiment.py --out-dir artifacts/answer_repair/v2 --bundle-name answer_repair_v2_gpu
```

生成 `configs/splits/answer_repair_v1_dev.json` 和 `artifacts/answer_repair/v1/answer_repair_v1_gpu.tar.gz`。解压后按包内 `README_GPU.md` 操作即可，无需 clone、下载原始数据、重建检索库或手动拼接未提交代码；模型首次运行仍需下载到 Hugging Face cache。

样本只取 Diagnosis v1 与 v2-expanded 的训练 ID 交集，并排除两版全部 validation/test IDs。v1 的 90 条 NQ train IDs 已在 v2 变成 validation/test，因此不能继续把整个 v1 train 当作 E 的开发池。交集内可选 FEVER 331 条、NQ 214 条。用固定 seed 42 的 SHA-256 排序，各选择 50 条，不使用 correctness、参考答案或 diagnosis 标签选择样本。参考答案仅保留用于原有离线评分。

首轮 FEVER 开发集初始正确 45、错误 5；NQ 初始 EM 正确 20、错误 30。两套 smoke 各 10 条，是各自开发集的前 10 条，不能与开发结果相加当成独立样本。先运行 FEVER smoke 并审核输出，再运行两套开发集的 Always-Verify 与等输出预算 Reanswer 对照。这一小规模结果只用于开发和判断是否继续扩展，不能作为最终 held-out 结果。

GPU 环境需要 Python 3.11+ 及与宿主兼容的 CUDA PyTorch。已有 CUDA PyTorch 时，其余 E 依赖只需 `requirements-answer-repair.txt`，不需要 FAISS 或 sentence-transformers：

```bash
python -m pip install -r requirements-answer-repair.txt
python scripts/check_answer_repair_environment.py --require-cuda --out environment.json
```

环境检查不会加载或下载模型。真实调用日志额外记录 Python、依赖版本、CUDA、GPU、模型配置中可取得的 revision、tokenizer revision 和 device map；revision 若为空，需要另行核实实际模型快照。断点续跑时环境记录发生变化会拒绝继续写同一文件。便携包附有所有输入与代码文件的 SHA-256 校验清单。

### 从仓库运行

也可以在 GPU 机器 clone `feature/verification-answer-repair` 分支，按 [SETUP.md](SETUP.md) 安装 CUDA PyTorch、项目依赖及正确的 Python 版本。使用同一个冻结 7B、NF4 和 greedy；不接入离线 70B backend。冻结输入包、结果和新 clone 的 CPU 回放步骤见 [交接清单](ANSWER_REPAIR_HANDOFF.md)。

下面展示 archived C1 的通用调用方式。首轮开发实际使用便携包中的训练交集样本；未经核对划分，不使用整个 500 条作为调 prompt 的开发集。工程 smoke 的结果使用独立文件：

```bash
python scripts/run_answer_repair.py --dataset fever --source-ref origin/feature/-ce-70B --all --limit 10 --out results/answer_repair/fever_smoke10.jsonl
```

检查 JSON 解析、引用校验、上下文长度和输出 token。正式运行：

```bash
python scripts/run_answer_repair.py --dataset fever --source-ref origin/feature/-ce-70B --all
python scripts/run_answer_repair.py --dataset nq --source-ref origin/feature/-ce-70B --all
```

接入 diagnosis 后，另开输出文件：

```bash
python scripts/run_answer_repair.py --dataset fever --source-ref origin/feature/-ce-70B --controller-decisions path/to/controller_decisions.jsonl --out results/answer_repair/fever_routed.jsonl
```

每次 `--limit` 表示固定前 N 条的开发 cohort；改变 N 必须另开输出文件。正式比较需要使用同一份 ID 划分，不能只从较易修复的失败中挑样本。

默认修复输出预算为 384 tokens，因 JSON 和短引用需要额外输出空间；C1 原有 64-token 设置及结果不修改。另有 `--strategy reanswer`，用同一证据及 C1 instruction 再次直接生成，作为额外调用/输出预算控制；它使用与 E 相同的 384-token 上限。Greedy 下相同 prompt 的输出常会保持相同，差异需要结合截断、模型 revision 和环境检查，不能自动归因于修复算法。

共享 model loader 的软件环境、远端 model revision 和 GPU 特性仍影响复现。运行时需保存 Git 提交、Python 包版本、模型 revision、GPU 和时间。脚本指纹防止不同输入、代码及设置混写，不能替代这些环境记录。

## 结果与成本解释

E 结果写入 `results/answer_repair/`，`condition=E`；proposal 的 `C2` 专指 accessibility-controlled evidence，不复用这个名称。每条保存初始/最终答案、证据 IDs、实际引用、原始输出、解析状态、额外调用和 token/延迟，source 指向冻结 C1 输入。

引用异常或证据不足均保留原答案，因此状态不是“修复成功”。离线评分单独读取参考答案，采用已有 NQ EM/F1 和 FEVER scorer；它不参与模型 prompt 或动作路由。

`.summary.json` 汇总全体评测样本，包括被路由跳过的样本：

- `fixed`：初始错误 → 最终正确；`regressed`：初始正确 → 最终错误。
- `net_correct_gain = fixed - regressed`；`accuracy_gain` 是其占样本总数的比例。
- `repair_success_rate` 的分母是初始错误数，`regression_rate` 的分母是初始正确数。
- 全体与变化样本计数、状态分布、NQ F1、配对 McNemar exact p-value。
- 额外调用、token 总量与额外生成延迟；`output_tokens_estimate` 来自输出文本重新分词，不包含 EOS。

额外生成延迟不含模型加载和整个框架耗时。没有 GPU 单价时不虚构货币成本。Archived C1 可能来自不同环境，不能直接用它的 latency 推导同一部署的端到端加速比；完整系统评测需要同环境计时，也要计 diagnosis/NLI/Probe 的成本。

## 续跑

结果每条 append、flush、fsync。输出锁保证同一文件只能有一个写入进程。中断后用同一命令继续，会检查指纹、ID 顺序、初始答案和评测一致性，再跳过已完成记录。

完整末条缺换行会自动补齐；不完整 JSON、重复/乱序结果、不同来源或不同 prompt/token 设置会拒绝续跑并保留原文件。硬中断可能遗留 `.lock`，确认原进程已停止后才移除。实验条件或输入改变时另开文件。

## 实验完成标准

按 2026-10-14 最新指令，人工结果复评可暂缓；待审核项已隔离，自动工程工作继续。工程完成不意味着算法有效性或全团队 held-out 验收完成。

首轮先在固定、同时包含正确和错误答案的开发集上报告 C1 与 Always-Verify 配对结果，审核所有 changed 和 invalid outputs，再决定是否改 prompt。人工审核应同时覆盖改对、改错、模型自称 keep、证据不足和 quote 合法却不支持结论的案例。

NQ 主表继续保留冻结 EM/F1；语义复评必须对初始和最终答案用同一 judge/人工规则，不允许只宽松评分修复答案。独立 verifier/人工 faithfulness 结果另报。

之后在 held-out 样本比较 Always-Verify 与 diagnosis-routed E，并与其他两个补偿操作形成 failure-type × action 矩阵。最终是否保留 E、怎样触发 E，应由净收益、误修率和成本结果决定；当前工程测试不提供这一效果结论。
