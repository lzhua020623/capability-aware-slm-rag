# E v7 实验结果：2026-10-14（Kaggle T4 x2）


协议见 [ANSWER_REPAIR_V7.md](ANSWER_REPAIR_V7.md)，运行前冻结。结果 ZIP `answer_repair_v7_reviewed_outputs_export.zip`（SHA-256 `48e82e600f189b2b5e7f98063883ffa3b14574c26debfc50d011170f7308d5d1`），原始 38 文件在 `artifacts/answer_repair/v7_reviewed/kaggle_20261014/`。云端 102 项测试通过，输入 56 文件校验通过；本地用冻结包代码回放全部 541 条记录（536 个唯一 ID）一致，见 `release_audit/cpu_replay.json`。环境：Python 3.13.15、torch 2.11.0+cu128、transformers 5.16.1、bitsandbytes 0.50.2，模型版本按协议固定。

**结论先行**：FEVER 的主体实体约束在开发批次上保住了离线回放的收益（审核后 4→10、dev50 45→46，无误修），但在 267 条 held-out 上净 −3（1 修复 / 4 误修）；NQ 的短答案策略在所有批次上冻结 EM 净负或持平，held-out 上显著为负（71→62，McNemar p=0.0225），且改动里有真实的语义误修。**两种策略都不晋升。** held-out 已按协议只跑一次，不再在这些样本上调整。

## 冻结指标

| 批次 | n | 初始正确 | v6r1 最终 | v7 最终 | v7 修复 / 误修 | v7 改答案 |
|---|---:|---:|---:|---:|---:|---:|
| FEVER 审核后 | 14 | 4 | 7 | 10 | 6 / 0 | 6 |
| FEVER dev50 | 50 | 45 | 44 | 46 | 1 / 0 | 1 |
| FEVER held-out | 267 | 232 | 无 | 229 | 1 / 4 | 5 |
| NQ 审核后（EM） | 14 | 2 | 0 | 3 | 1 / 0 | 2 |
| NQ dev50（EM） | 50 | 20 | 7 | 19 | 2 / 3 | 9 |
| NQ held-out（EM） | 146 | 71 | 无 | 62 | 2 / 11 | 20 |

NQ F1：审核后 0.292→0.368，dev50 0.549→0.512，held-out 0.632→0.586。审核后与 dev50 的 FEVER 结果与设计阶段的离线回放逐条相同（同一分类器分数、确定性规则）。

| 批次 | Qwen 调用 / 生成秒数 / 输入 tokens | NLI forward 批次 / 分类秒数 / 输入 tokens |
|---|---|---|
| FEVER 审核后 / dev50 / held-out | 0 | 29 / 12.6 / 21,910；101 / 30.6 / 62,352；537 / 170.1 / 337,180 |
| NQ 审核后 / dev50 / held-out | 14 / 75.3 / 11,517；50 / 293.4 / 45,385；146 / 878.5 / 132,829 | 0 |

## FEVER `fever_nli_anchored`

held-out 267 条按离线标签：232 条 correct（157 kept、64 uncertain、7 conflict、4 repaired = 4 条误修）；35 条 insufficiency（28 uncertain、6 kept、1 repaired = 唯一修复）。这批没有 utilisation 标注，所以它只能衡量“会不会改坏”：误修率 4/232 = 1.7%，弃判率 92/267。

4 条误修都发生在通过了词法主体约束的窗口上。通过约束不等于实体消歧成功：其中既有 NLI 忽略限定词/属性类别的问题，也有主体只取到 sexual、市与县未区分的问题：

| ID | 声明 | 判定 | 原因 |
|---|---|---|---|
| 49192 | The Armenian Genocide **rarely** was the extermination… | REFUTES→SUPPORTS（E=0.976） | NLI 忽略了 rarely |
| 164986 | Polar bears are classified as endangered. | REFUTES→SUPPORTS（E=0.918） | 物种保护等级判断错误 |
| 14448 | Sexual slavery is a reason for human trafficking. | SUPPORTS→REFUTES（C=0.932） | 主体块只有 "sexual"，labor trafficking 页面被当作反驳 |
| 126292 | Camden, New Jersey is a county. | REFUTES→SUPPORTS（E=0.984） | Camden County 页面蕴含“是县”，市/县实体混淆 |

唯一修复 104922（Oscar Isaac / Apocalypse）引用 X-Men: Apocalypse 页面，窗口含 Oscar Isaac，是有据的。

## NQ `short_answer_quote`

状态分布：审核后 11 invalid / 2 repaired / 1 kept；dev50 32 / 9 / 9；held-out 99 invalid / 20 repaired / 25 kept / 2 unanchored。210 条里 142 条无效，原因：61 条模型把编号写成 `RANK: [1]`；72 条引文定位失败；8 条引文超 60 词；1 条答案超 20 词。引文定位失败里有引号包裹、尾词截断（ridge/ridges）、引错段落（thumb twiddling 引 R3 却标 R1）和真实的删句拼接（Columbia 案删掉 ", Columbia 's 28th mission"）。

**放宽解析的事后回放（不晋升，仅分析）**：`scripts/replay_short_answer_lenient.py` 用同一批原始输出，接受 `[n]`、去掉引号包裹、把 `ANSWER: INSUFFICIENT` 当弃答，结果见 `release_audit/lenient_parser_replay.json`：

| 批次 | 冻结 v7 EM | 放宽解析 EM | 放宽后修复 / 误修 |
|---|---|---|---:|
| 审核后 14 | 2→3 | 2→3 | 1 / 0 |
| dev50 | 20→19 | 20→20 | 3 / 3 |
| held-out | 71→62 | 71→58 | 4 / 17 |

本次事后回放中，放宽解析增加了有效输出，也增加了误修，说明只修格式不足以恢复净收益。模型看不到初始答案、重新选择答案可能造成不必要替换；这仍是待消融验证的原因假设，不能仅凭当前对照认定初始答案可见就一定更安全。

held-out 146 条按离线标签：126 条 correct 里 11 条 EM 误修（84 invalid、23 kept、17 repaired、2 unanchored）；7 条 accessibility 全部 invalid；13 条 insufficiency 改了 3 条、无一变对；55 条“标注正确但 EM 判错”的样本只改了 6 条、2 条变成 EM 正确。这批没有 utilisation 标注，不能可靠估计目标错误层的修复率，主要用于检查误路由风险；缺少该标签不等于可以证明没有任何真实 utilisation 错误。

held-out 20 条改动的代理初读（待人工，见 `release_audit/review_priority.md`）：
- 真实误修（答案类型或对象错了）：spirits 词源 Middle Eastern alchemy→"spirit of the original material"；GoT 祖母扮演者 Rigg→角色名 Lady Olenna Tyrell；南非青年节 1976→"16 June"；Only Love Can Break Your Heart 的 Neil Young→翻唱者 Saint Etienne；冰壶 1998→1924 有争议。
- EM 假退步（内容没错、只是变长或加了介词/前缀）：Himadri Station→Himadri、"On September 9, 2012"、Road/Track 的空格、"in southern Turkey"、"off Flamborough Head"、Paige O'Hara 加前缀、The Crossing 更具体的地点、心脏位置更具体。
- EM 修复：SA node→the sinoatrial node、7→seven（均为写法对齐，不是语义修复）。

dev50 的 9 条改动中，Greenland→"an autonomous constituent country within the Kingdom of Denmark"（问大洲）和 Feed Jake→歌词片段（问原唱）是不回答问题的真实误修；War of 1812 把双方都列出来；其余为 EM 假退步或假修复。审核后 14 条里 Annie 演唱者是真实修复（v6r1 已确认），bias 定义只是整句化。

## 这轮说明了什么

1. FEVER：主体约束在开发批次得到 6/0、1/0，但 held-out 的标签误修率为 4/232 = 1.7%；这是整个 E 决策链对初始正确答案的误修比例，不是 NLI 在实体正确窗口上的通用错误率。冻结 Controller 在 PRIMARY test 上不把 FEVER 路由到 E，FEVER 单模块结果仍用于 Always-Verify 和失败类型 × 操作矩阵，不能代替自适应路径收益。
2. NQ：候选优先的短答案策略在第一批 E 未见数据上冻结 EM 净负，且代理初读有真实语义误修。v4 dev50 的 20→20 来自 grounded_reanswer / reanswer；grounded_reanswer 同样隐藏初始答案，不能作为 verify-first 更安全的证明。下一轮可以测试“先核验原答案，再最小修复”，但必须单独消融。新测试还需补充未使用的目标错误，不能只靠剩余 1,287 条 correct/insufficiency 样本估计修复率。
3. 下节的 `heldout3216` 已完成一次运行，提供 utilisation 标签层的修复信号，但不能视为可靠修复率的上界。它按标签分层，标签和语义输出仍需核验，也不代表部署分布。

## 人工审核队列

`release_audit/review_priority.md`：43 条改动结果优先（18 条冻结指标退步、13 条改善、12 条其他改动），其余 498 条保留答案。沿用确认清单的 7 条完整性口径；代理初读不是人工真值。

## 第 4 批 `heldout3216`：含 utilisation 标注的 NQ held-out（只跑一次）

结果 ZIP `answer_repair_v7_heldout3216_outputs_export.zip`（SHA-256 `91ee085bb38c8ee05220b71e5b547978871d84635261d8034a177370a8b927fd`），原始 11 文件在 `artifacts/answer_repair/v7_heldout3216/kaggle_20261014/`；输入 50 文件校验通过，云端 102 项测试通过，本地冻结代码回放 300 条一致。300 次 Qwen 调用，生成 1618.4 秒，输入 272,962 tokens，输出估计 18,980 tokens。

这是唯一一批 E 从未见过、又含可修复样本的数据（按离线标签分层选取，不代表部署分布）。整体冻结 EM 63→67（13 修复 / 9 误修，McNemar p=0.52），F1 0.304→0.334；按层：

| 层 | n | EM 前→后 | 修复 / 误修 | 改答案 | invalid | F1 前→后 |
|---|---:|---:|---:|---:|---:|---:|
| evidence_utilisation | 120 | 0→10 | 10 / 0 | 24 | 83 | 0.063→0.146 |
| evidence_accessibility | 35 | 0→3 | 3 / 0 | 9 | 18 | 0.076→0.164 |
| correct | 100 | 63→54 | 0 / 9 | 14 | 56 | 0.801→0.759 |
| evidence_insufficiency | 45 | 0→0 | 0 / 0 | 5 | 35 | 0.018→0.023 |

utilisation 层的 10 条 EM 修复（Tamara Drasin、Billy Idol、Davos、Astros 2017、Hasmukh Adhia、Planet of the Apes 7 月 14 日、脱欧公投 6 月 23 日、AGT 第 13 季、红卫兵 1966–1967、十字架 4 世纪）都是把错误实体/日期换成证据里的正确值；另有 elevator pitch（F1 0.09→0.90）、迪士尼乐园地点（→0.89）、Bastard Executioner "one season"、贫困率 13.5% 等 EM=0 但明显改善的案例，代理初读下 utilisation 层的语义修复约 13–14/120。改错的方向也有：Spock 手势→犹太祭司祝福（证据里的词源，参考要 Nimoy）、hairpin、criminal law、Home Alone 2 New York→New York City（参考 Florida）均为错→错，不计误修。

correct 层 9 条 EM 退步里，代理初读认为 7 条是加字/加日期/空格（"Hal ( Bryan Cranston )"、"Burj Khalifa in Dubai…"、"March 18, 2005"、"the volume of a gas"、"more in the wintertime"、"1980 summer Olympics"、"pulse - width"），1 条有争议（"RMS Titanic"→"Titanic"，题问全名），1 条是真实误修候选（菲律宾最老街道 Colon Street→费城 Elfreth's Alley，实体错误）。63 是初始 EM 正确数，不是经人工确认的语义正确数；正式语义误修率需审核初始/最终语义正确性后重新确定分子与分母。

放宽解析的事后回放（`release_audit/lenient_parser_replay.json`，不晋升）：整体仍 63→67（16 修复 / 12 误修）；utilisation 0→11，correct 63→53。和主包 held-out 一致：放宽格式只多换来 1 个修复，多付 3 个 EM 误修。192 条无效输出里 126 条是引文非逐字、53 条 `RANK: [n]`、13 条超长。

**这一批改变什么**：按上游 utilisation 标签，v7 有 10/120 = 8.3% 的冻结 EM 修复；代理初读还有若干 EM 未识别的改善，但语义约 11% 仍是待人工核实的估计。另一批 held-out 的演员→角色、原唱→翻唱、年份→日期等误修候选说明正确答案保护仍需改进。不能把两批初始 EM 正确数 71+63=134 直接当作语义误修率的分母，也不能把代理估计写成正式人工指标。结合误路由风险，目前仍**不晋升**。下一版测试原答案可见的核验与最小修改；剩余 1,287 条可作安全性材料，但不足以独立验收目标修复能力。

人工审核队列：`artifacts/answer_repair/v7_heldout3216/release_audit/review_priority.md`，52 条改动优先（9 条 EM 退步、13 条改善、30 条其他）。

## 结果复核、数据决策与下一轮路线（2026-10-14）

本节在 `5295edc` 的两批归档上复核。主包 ZIP 的 38 个文件与入库解压文件逐字节一致，冻结代码回放 541 条；第 4 批 ZIP 的 11 个文件也一致，回放 300 条。合计 841 条记录、836 个唯一 dataset/sample_id，95 条改动记录对应 93 个唯一样本。两批 pair_review 全部 pending，CPU 回放均明确 semantic_correctness_verified=false。主包与第 4 批没有重新执行模型推断。

### 当前不足来自哪里

- 方法：NQ 从头选择新答案后，缺少“新答案为什么比原答案更可信”的验收；引文来源与弱词法相关性不足以保证答题对象、事件、时间、答案类型及完整性。FEVER 的大写主体规则仍会提取不完整主体，不能区分 Camden 市/县；分类器也会忽略 rarely 等限定词。
- 工程：主包 NQ 142/210 无效，第 4 批 192/300 无效（64%）。格式修复值得做，但事后放宽解析并未消除净损失，不能把它当作唯一根因。
- 数据构成：第一批 held-out 没有 utilisation 标签；它可以揭示误改，却无法估计目标层的修复率。第 4 批才补入 120 个 utilisation 标签样本。
- 标签质量：第 4 批 300 条的上游 annotation_source 全为 ai_initial_nq3216_2026-10-09。needs_manual_review=false 不能解释成人工验收。工作簿 gold/EM 的来源一致性已检查，但这不保证每个参考答案及失败类型正确。
- 评测独立性：两批 held-out 是 E 当时未使用的数据，仍来自 Diagnosis 的训练范围；不能把它们称为整个 Diagnosis→Controller→E 系统的独立测试。现已读取结果，后续改策略时应把这些批次归为已见分析材料。

### 扩充、增强还是更换

建议保留 NQ/FEVER 主任务，定向扩充经核验的目标错误、正确对照及困难边界。更多评测记录不会自动改变当前未微调的 Qwen/NLI 权重；它们主要改善误差分析、校准与统计可信度。没有证据支持为了获得正结果而更换主数据集。

复核上游 v2-expanded TRAIN、排除 preliminary500 后共有 1,587 个 NQ ID，本轮选择 300；剩余 1,287 个按现有标签为 correct 1,164、insufficiency 123，**utilisation/accessibility 均为 0**。全部 120 个 utilisation 和 35 个 accessibility 已用于本轮，剩余池不能单独作为下一轮修复率测试。需要从尚未用于 E 开发的新 NQ/FEVER ID 重建同一 C1 条件并核验失败类型，或由团队预先规划全新的共同测试集。先按 ID/来源分组冻结开发、校准与测试，再查看测试结果。

数据增强可用于开发压力测试，或在未来明确训练核验器/接受策略时使用：同名实体、演员/角色、原唱/翻唱、事件年/报道日、否定与限定词、数量和范围扰动。每个合成标签需确认，按原始来源分组划分，合成样本不能冒充独立自然测试样本。

可选外部补充是 [RAGTruth](https://github.com/ParticleMedia/RAGTruth)：官方提供多模型 RAG 回答及词级幻觉人工标注，适合额外测试核验器。它的模型输出、任务与本项目固定 Qwen 的 C1 不同，不能直接替换端到端主评测。若核验器本身在 RAGTruth 上训练，应使用未参与训练/调参的官方测试范围，并说明已知训练来源。

### 下一轮候选路线：统一流程、受控比较，待重新开发验证

两任务采用同一逻辑：问题/声明 + 现有 C1 + 初始答案 → 核验原答案 → 必要时由固定 Qwen 作最小修复 → 核验候选及答题完整性 → 接受或回退。FEVER 与 NQ 的输入表述和输出指标分别适配；核验器作为可替换组件，避免把“数据集变了”与“模型变了”同时作为有效性的依据。保留 v7 直接 NLI 标签决策为一个历史基线，额外模型预算单独记录。

1. 先确认本轮关键输出和分层标签。优先 93 个唯一改动案例，再抽审未改/无效输出；不能只审 EM 退步。分别标初始/最终正确性、证据充分性、引用支持及问题匹配，保留参考问题和边界原因。
2. 在已见开发材料上建立正误与边界核验集。先比较当前 MNLI、proposal 候选 `cross-encoder/nli-deberta-v3-small`、MiniCheck 的原答案误报率/漏报率及成本，校准弃判区间。阈值 0.90 不应跨模型直接照搬。
3. 做最少量消融：不改答案、v7 候选优先、原答案可见的核验后修复、同一修复方案加独立核验。输入、固定 Qwen、推断设置和成本口径一致，一次只改变一个因素；格式解析与语义接受规则分开评估。原答案可见可能锚定旧错误，需同时报告漏修。
4. 只有明确核验出原答案的问题、候选完整回答查询且证据更充分时才替换。相同实体词不能代替实体与关系匹配；不支持不等于反驳，不能将二分类核验器的 unsupported 直接映射到 FEVER REFUTES。
5. 冻结选定方法后在新的目标错误和正确对照上各跑一次，再由团队在共同未见测试集测 Controller 选择下的净收益、误修、覆盖和完整成本。E 的实际收益取决于进入 E 的可修复错误比例与正确样本误路由比例，不能由分层 300 条的总体 EM 外推。

样本量按目标确定：在独立二项抽样假设下，若 300 个已确认正确对照零误修，真实误修率的单侧 95% 上界才约 1%（1−0.05^(1/300)）。这只是说明稀有误修需要较多正确对照，不是当前实验已经达到的保证；修复差异仍需按预期配对改善做功效规划。

### 已检索的实现参考

| 实现 | 可借鉴内容 | 本项目使用建议与限制 |
|---|---|---|
| [MiniCheck](https://github.com/Liyan06/MiniCheck) | 文档—声明的句级支持判定，公开小于 1B 的 Flan-T5-Large 版本及合成数据方法 | 优先作为核验器对照；其输出是 supported/unsupported，不能直接三分类，也不能自动保证候选回答了原问题。见 [官方模型卡](https://huggingface.co/lytang/MiniCheck-Flan-T5-Large)。 |
| [LettuceDetect](https://github.com/KRLabsOrg/LettuceDetect) | 输入 question/context/answer，定位答案中不受支持的片段 | 可用于定位后最小修复；问题匹配、答案完整性与漏检仍需单独测量，不能以无幻觉片段直接等同答题正确。 |
| [RARR](https://github.com/anthonywchen/RARR) | Agreement Gate 后才编辑原回答，尽量保留原内容的修复思路 | 优先参考门控和编辑流程；原实现包含额外搜索和 API 模型，本项目应在固定 C1/固定 Qwen 下实现受控变体并记录全部成本。 |
| [AlignScore](https://github.com/yuh-zha/AlignScore) | 文本对事实一致性评分与分块核验 | 可作第二阶段评分对照；官方实现的依赖较旧，应隔离环境并核查训练来源，不将论文榜单数值当成本项目效果。 |

以上是官方仓库/模型卡核对后的候选与设计参考，未在本项目 GPU 上测试或晋升。首轮只需当前核验器、一个轻量 NLI 和一个文档级核验器作对照，按误修风险、修复覆盖和成本决定是否更换。
