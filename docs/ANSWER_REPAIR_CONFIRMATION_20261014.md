# E v6r1 关键人工确认清单（2026-10-14）

> **状态：已确认（2026-10-14，用户回复“同意”）。** A、B 按两读结论确认；C 按下文建议改判（113、121 暂缓）；D 按记录方式处理。记录见 `data/annotations/human/answer_repair_v6r1_confirmations_20261014.json` 与 `answer_repair_v6r1_pairs_20261014.jsonl`；去重后语义修复 9、误修 3、暂缓 2。其余 97 个唯一样本待批量确认。

给 E 负责人逐条确认用。编号沿用 `artifacts/answer_repair/v6r1_reviewed/assisted_review_20261014/assisted_review.md` 的 1–128。每条给出两份独立读法：Codex 辅助审核（2026-10-14，参考可见）和独立盲评（Claude，2026-10-14，只看问题、初始/最终答案、模型实际引用、Top-5 原文和冻结参考，**没有看 Codex 结论**）。两读都不是人工真值；确认后用 `scripts/apply_answer_repair_confirmations.py` 写入人工记录，再汇总语义修复/误修。

两读在 28 个关键案例的四项布尔上有 22 条完全一致；6 条差异全部落在 Codex 标为“待定”的字段上（盲评给了明确判断），以及 #30 的“初始是否有支持”（合理拒答算不算有支持）。没有任何一条两读方向相反。

## 统一的答案完整性口径（建议采用）

1. **正确** = 回答了问题要的对象、关系、时间和范围。同义短答和包含正确值且没有错误主张的长句都算对；长句不因 EM=0 判错，短答不因参考更长判错。
2. **定义题**（what is X / what does X mean）要给出定义谓词，只重复名称不算对；省略次要限定词（如 measurable）仍算对，但记 `completeness_boundary`。
3. **who/when/where** 必须给出具体实体/日期/地点；只给相对位置或相关事实不算对。
4. **latest/last/originally** 按冻结语料中明确标注日期的目标事件判断，不按当前知识。
5. **拒答**（not specified 等）不算正确答案；证据确实缺失时记为“有依据的拒答”，但计入未修复。
6. **FEVER**：SUPPORTS 要求整条声明可由证据推出，REFUTES 要求证据与声明矛盾；证据沉默时按冻结标签暂定并标 `reference_dependent`，不算独立事实确认。
7. **支持性**与正确性分开判：真实但不回答问题的句子不算支持；模型实际引用与 Top-5 是否足够也分开记。

## A. 修复候选（9 个唯一样本）：建议全部确认为语义修复

| 编号 | 数据集 / ID | 问题 | 初始 → 最终 | Codex | 盲评 | 备注 |
|---|---|---|---|---|---|---|
| 18 | FEVER 156493 | Daggering is not a type of dance. | SUPPORTS → REFUTES | 修复 | 修复 | R1 首句 "is a form of dance" |
| 19 (=21) | FEVER 203615 | Sugarland Express co-written by Spielberg | REFUTES → SUPPORTS | 修复 | 修复 | 两批重复，只算 1 次 |
| 20 | FEVER 72578 | Greek has speakers in Albania | REFUTES → SUPPORTS | 修复 | 修复 | |
| 22 (=33) | NQ tibia/fibula | where is the tibia and fibula bone located | on the lateral side → "The fibula ... is a leg bone located on the lateral side of the tibia ..." | 修复（中置信） | 修复（中置信，胫骨位置只是隐含） | 唯一需要你拍板的：按口径第 3 条，初始只给相对位置算错；最终给出 leg。建议算修复 |
| 23 (=35) | NQ Annie | who sang oh annie i not your daddy | 歌名 → Kid Creole and the Coconuts 整句 | 修复 | 修复 | EM 漏掉 |
| 25 (=38) | NQ Loihi | what is loihi and where does it originate from | 名称 → 海底火山 + 夏威夷东南 | 修复 | 修复 | 口径第 2 条 |
| 27 | NQ serrated adenoma | what is a serrated adenoma of the colon | 病名 → 结肠癌前扁平病变 | 修复 | 修复 | 口径第 2 条 |
| 28 | NQ Astros | who did the astros play last time they were in the world series | White Sox (2005) → Dodgers (2017) | 修复 | 修复 | 口径第 4 条 |
| 29 | NQ Jenny | when does jenny come back to gossip girl | 报道日期 → 第六季最终集 | 修复 | 修复 | 初始把报道日期当回归时间 |

## B. 误修候选（3 个）：建议全部确认为语义误修

| 编号 | 数据集 / ID | 问题 | 初始 → 最终 | Codex | 盲评 | 模型引用 |
|---|---|---|---|---|---|---|
| 1 | FEVER 140820 | Men in Black II is a film from America. | SUPPORTS → REFUTES | 误修 | 误修 | 引用 2015 年比利时电影 Black，实体错误 |
| 2 | FEVER 33982 | Henry Cavill starred in two Superman films. | SUPPORTS → REFUTES | 误修 | 误修 | 引用 Henry's Crime，实体错误 |
| 42 | NQ statute | a legislative act passed by congress is an example of | statute → "The act was passed by large majorities in both houses of Congress." | 误修 | 误修 | 引用逐字成立但不回答问题；冻结 EM 前后都是 0，被指标漏掉 |

两条 FEVER 误修在 v7 的主体实体约束下离线回放均不再触发（见 `docs/ANSWER_REPAIR_V7.md`）。

## C. 未决项（7 个）：建议的默认处理

| 编号 | 问题 | 争点 | Codex | 盲评 | 建议 |
|---|---|---|---|---|---|
| 31 | what is property of a system in thermodynamics | 初始省略 "measurable" 是否仍算对 | 初始待定 | 初始正确 | 按口径第 2 条判**初始正确**，前后均对，不算修复 |
| 32 | what is the basic principle of christian morality | 单字 "love" 是否足够 | 初始待定 | 初始正确（口径边界） | 判**初始正确**，不算修复；记 completeness_boundary |
| 54 | The Mighty Ducks was only produced by a Canadian film producer. | C1 是否足以反驳；引用的是续集 D3 | 支持待定 | 支持 = 是（低置信，partial） | 前后 REFUTES 正确；支持性按**不足**记（引用续集，制片人国籍缺失），不影响修复计数 |
| 109 | what does hp mean in war and order | 证据只给通用含义，未提该游戏 | 支持待定 | 支持 = 是 | 前后正确；支持性判**是**（通用定义足以回答缩写含义） |
| 113 | little league 投手距离 | Top-5 只有 MLB 的 18.44 m，参考本身也是 MLB 数值 | 正确性待定 | 正确性 null | **暂缓**，标 reference_mismatch；不计入修复/误修 |
| 121 | when did the soviet union start and end | 参考 "in Eurasia" 答非所问；1991 不在证据里 | 正确性待定 | 正确性 null | **暂缓**，标 reference_mismatch；不计入 |
| 125 | who sings i just want to use your love tonight | 证据没有这句歌词，需要歌词→歌名推断 | 支持待定 | 支持 = 是 | 前后正确；支持性判**是**（歌名 Your Love 与歌词对应是常规推断） |

## D. 参考错配与口径案例：建议处理

| 编号 | 问题 | 两读一致结论 | 建议 |
|---|---|---|---|
| 30 | what degree is a crock pot on low | 前后均错；参考 71–74 °C 对应 warming 档；最终答案是无关句子 | 记 reference_mismatch + nonresponsive_final；初始拒答按口径第 5 条不算对。盲评认为初始无支持（拒答不算支持），Codex 认为有依据；按“拒答有依据但未回答”记录即可 |
| 41 | 教会用哪个节日取代 Samhain | 前后均错；最终引用 12 月 25 日段落，事件错误；正确值不在 Top-5 | 记 wrong_event_citation，不算修复 |
| 36 | UNC 上次缺席 NCAA | 前后均错；最终谈场地抵制，事件错误 | 同上 |
| 101 | paleomagnetism ... as demonstrated by | 两读均判前后错（答案复述前提，没给岩层记录） | 维持你之前对旧第 17 条的修正理由 |
| 103 | 药学院白袍 | 两读均判前后正确；参考 "before the first year" 是泛指医学院 | 维持旧第 23 条确认 |
| 104 | who was originally cast to play indiana jones | 两读均判前后错（证据明确 Tom Selleck 被选中，Ford 只是建议），证据清楚却未修复 | 记 missed_repair，是 v7 要重点看的案例 |
| 3 / 24 / 26 | Ringo Starr / bias / Columbia | 前后均对，最终只是有据扩写 | 作为 EM 假退步的代表案例；NQ 其余 12 条 EM 退步同理（5–17 号） |

## E. 需要你回复的内容

按编号回复即可，例如：

> A 全部同意；B 全部同意；C：31、32 初始算正确；54 支持不足；109、125 支持是；113、121 暂缓；D 同意。

回复后执行：

```sh
python3.11 scripts/apply_answer_repair_confirmations.py --confirmations <你的确认.json> --out data/annotations/human/answer_repair_v6r1_pairs_20261014.jsonl
```

它把确认写到每条记录的 `pair_review`（含重复记录），绑定结果 SHA-256，并输出按批次的语义修复/误修计数。按上述建议，去重后的 v6r1 语义结果将是：**9 个修复、3 个误修、2 个暂缓**；其余 109 条保持辅助审核结论、待后续批量确认。
