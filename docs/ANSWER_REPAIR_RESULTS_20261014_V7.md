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

4 条误修都发生在通过了主体实体约束的窗口上，也就是说锚点规则挡住的是同名页面，挡不住 NLI 本身的判断错误：

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

有效输出越多，误修越多。所以无效输出不是收益被格式吃掉，而是在替正确答案挡刀；问题在策略本身：模型看不到初始答案、从头选答案，本来正确的答案有相当概率被换成另一个片段。

held-out 146 条按离线标签：126 条 correct 里 11 条 EM 误修（84 invalid、23 kept、17 repaired、2 unanchored）；7 条 accessibility 全部 invalid；13 条 insufficiency 改了 3 条、无一变对；55 条“标注正确但 EM 判错”的样本只改了 6 条、2 条变成 EM 正确。这批同样没有 utilisation 标注，修复无从发生；它衡量的是害处。

held-out 20 条改动的代理初读（待人工，见 `release_audit/review_priority.md`）：
- 真实误修（答案类型或对象错了）：spirits 词源 Middle Eastern alchemy→"spirit of the original material"；GoT 祖母扮演者 Rigg→角色名 Lady Olenna Tyrell；南非青年节 1976→"16 June"；Only Love Can Break Your Heart 的 Neil Young→翻唱者 Saint Etienne；冰壶 1998→1924 有争议。
- EM 假退步（内容没错、只是变长或加了介词/前缀）：Himadri Station→Himadri、"On September 9, 2012"、Road/Track 的空格、"in southern Turkey"、"off Flamborough Head"、Paige O'Hara 加前缀、The Crossing 更具体的地点、心脏位置更具体。
- EM 修复：SA node→the sinoatrial node、7→seven（均为写法对齐，不是语义修复）。

dev50 的 9 条改动中，Greenland→"an autonomous constituent country within the Kingdom of Denmark"（问大洲）和 Feed Jake→歌词片段（问原唱）是不回答问题的真实误修；War of 1812 把双方都列出来；其余为 EM 假退步或假修复。审核后 14 条里 Annie 演唱者是真实修复（v6r1 已确认），bias 定义只是整句化。

## 这轮说明了什么

1. FEVER：主体实体约束按设计工作，开发批次 6/0、1/0；但 NLI 在实体正确的窗口上仍有 1.7% 的误判率，而 Controller 在 PRIMARY test 上本就不把 FEVER 路由到 E。FEVER E 的价值限于 Always-Verify 基线和失败类型 × 操作矩阵，在那里 1.7% 的误修要和 Always-Verify 的收益一起报。
2. NQ：候选优先的短答案策略在 E 从没见过的数据上净负，且有真实语义误修。这和 v6r1 的 15 条 EM 假退步不是一回事：这次 held-out 的 11 条 EM 退步里约 5 条是真的改坏。v4 的 verify-first（初始答案可见、只在有更强证据时替换）在 dev50 上曾是 20→20，比候选优先更安全；若做 v8，应回到 verify-first 并用 3216 池剩余的 1,287 条未用 ID 做新的 held-out，不能再用本轮三批。
3. 唯一还没测的数字是 NQ 在 utilisation 样本上的修复率。`heldout3216`（300 条，含 120 条 utilisation）包已冻结未跑；按协议它只跑一次，跑不跑由你决定。考虑到 held-out 的误修率，它的价值是给出“最好情况下的修复率上限”，而不是决定晋升。

## 人工审核队列

`release_audit/review_priority.md`：43 条改动结果优先（18 条冻结指标退步、13 条改善、12 条其他改动），其余 498 条保留答案。沿用确认清单的 7 条完整性口径；代理初读不是人工真值。
