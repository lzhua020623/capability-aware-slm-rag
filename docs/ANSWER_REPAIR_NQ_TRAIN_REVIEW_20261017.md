# NQ Train 开发样本辅助审核：2026-10-17

截至 2026-10-19 第六批意见，固定首批 **100 条均已有一轮人工意见；75 条定案（11 条 E utilisation 目标、64 条正确对照），25 条待定，未过目为 0**。第一轮过目完成不等于所有主标签定案，更不表示全量 74,097 条已标注。25 条包括最初 17 条暂缓、第四批 5 条完整性／表达口径分歧及 N58/N61/N83 条件或补充要求；用户原话与代理复核分别保留，不自动当作事实错误。64 条对照含 N76 的同实体来源表述核对，已单独标明。原始数据以 `origin/main@058f10baa47083220f95edb39db8f106aa7e85d6` 为准；此前核对时未发现该提交之后的新 main 提交，本轮未重新查询远端。

当前状态见 [累计 100 条审核记录](../data/annotations/human/answer_repair_nq_train_dev_v1_through_batch06_20261019.jsonl) 和 [累计计数及待定分组](../data/annotations/human/answer_repair_nq_train_dev_v1_through_batch06_20261019.summary.json)。前五批记录及 [第五批累计历史计数](../data/annotations/human/answer_repair_nq_train_dev_v1_through_batch05_20261019.summary.json) 保留原样。这些是查看辅助分析和证据后的对话意见，不是独立盲标；没有确认任何 E 新输出正确。

2026-10-17 接入时核对 main 新增的 11 个交付/流水线文件与本地版本，均逐字节一致；Google 下载的 15 个原始分片均与 main 清单的 SHA-256 一致。74,097 条源记录及其 gold 未修改。E 的转换文件、内部划分与审核建议均为派生产物；如与 main 后续正式协议或团队共同划分冲突，应按上游重建新的派生版本。

## 第六批人工意见与首轮收口：2026-10-19

用户明确确认 N86/N87/N88/N90/N92/N93/N95/N97/N100 均正确，9 条全部加入正确对照。原话及范围见 [第六批确认](../data/annotations/human/answer_repair_nq_train_dev_v1_batch06_20261019.confirmations.json)，逐条记录见 [第六批人工记录](../data/annotations/human/answer_repair_nq_train_dev_v1_batch06_20261019.jsonl)。全部来源记录、段落引用及字符位置校验通过；N87 的优势概述与 N88 的协议／规范表述只按本例接受，不推导全局简答规则。

| 首轮处理结果 | 数量 | 用途 |
|---|---:|---|
| 内部 E utilisation 目标 | 11 | 开发时检查是否修复原错误 |
| 正确对照 | 64 | 开发时检查是否误修已有正确回答 |
| 待定 | 25 | 隔离，不用于当前修复收益统计 |
| 尚未过目 | 0 | 100 条均已收到人工意见 |

当前可用开发输入为 [75 条清单](../configs/splits/answer_repair_nq_train_confirmed_dev_v6.json)、[冻结 C1 子集](../artifacts/answer_repair/nq_train_dev_v1/confirmed/through_batch06_c1.jsonl) 和 [纯运行时子集](../artifacts/answer_repair/nq_train_dev_v1/confirmed/through_batch06_runtime.jsonl)。75 条全部来自 development，校准／预留交集为 0，runner 冻结评分预检通过。25 条旧待定记录逐条及原字节内容保持一致，不因首轮结束自动解禁。gold、原 C1、冻结 scorer、上游 Diagnosis 标签、v7 结果和 E 算法均未改，未运行新 GPU 实验。

待定清单按来源分为三组：

- 原争议未解决 17 条：N3/N7/N19/N28/N44/N56/N57/N59/N63/N67/N78/N79/N82/N89/N91/N98/N99。
- 完整性／表达口径 5 条：N13/N18/N22/N41/N43。
- 购买条件、地点范围或时间限定 3 条：N58/N61/N83。

下一轮可使用 11 个目标和 64 个正确对照推进原答案核验、最小修复及候选核验的受控开发，25 条继续隔离，不要求先强行定类。需要分别记录改对、误修、无变化、解析失败及调用成本；该子集经过人审选择且不是未见测试集，不能直接外推总体增益。本轮只完成审核整理与输入准备，不代表 v8 已实现或有效。

网页最新导入文件为 [第六批累计状态](../artifacts/answer_repair/nq_train_dev_v1/review/NQ_train_pending_92.chat_confirmations_batch06_20261019.json)，原 92 条包显示 67 条确认、25 条待定、0 条未审；另 8 条最早确认的 E 目标本就不在该页。导入前先备份其他浏览器草稿，文件包含前几批意见，原数据包哈希不变。

## 第五批人工意见：2026-10-19（历史）

本批共 22 条，18 条明确判对：N51/N53/N54/N55/N60/N64/N65/N66/N68/N69/N70/N71/N72/N73/N75/N77/N80/N85。逐条引用位置、段落及来源哈希均通过核对；按冻结语料时点记录，N51 的 current 和 N80 的 new 均不解释为今天的最新版本或人物状态。

| 编号 | 用户意见与记录 |
|---|---|
| N58 | 用户将 18 判错，提出 21，原话及否定裁决保留。R1 同时写一般 non-NFA 购买门槛 18，以及向 federal firearms licensees 购买 handgun 为 21；题目未限定枪种／渠道，gold 为 18 years。记录购买条件争议，完整标签暂缓，不直接以 21 覆盖所有场景 |
| N61 | 用户希望写澳大利亚昆士兰州图文巴市，即 Toowoomba, Queensland, Australia。R1 明确校本部位置，但也支持 C1 列出的 Toowoomba/Springfield/Ipswich 三校区。记录完整地点和校本部／校区范围要求，暂不归为事实错误 |
| N76 | 用户说“荷兰的圣尼古拉斯”，按 R1 的 Dutch St Nicolas traditions 及 R2/R3 的 Dutch Sinterklaas 核对为同一来源表述，保留正确对照。该条标注为同实体解释，不冒称用户直接写了“对”；不将历史上的希腊主教说成荷兰人 |
| N83 | 用户要求时间限制，记录 R1 的“截至 2018 年 7 月，Instagram 自己的账户”。原实体在冻结语料口径下有支持；缺少时间说明是否判原答案错误尚未明确，先记录时间完整性要求，完整标签暂缓，不查询或替换为今天排名 |

本批意见完整保存于 [第五批确认范围](../data/annotations/human/answer_repair_nq_train_dev_v1_batch05_20261019.confirmations.json) 和 [逐条记录](../data/annotations/human/answer_repair_nq_train_dev_v1_batch05_20261019.jsonl)。本批为 19 个正确对照（18 个明确“对”及 N76 的等价来源说明），无新增 E 核心目标；N58/N61/N83 单列待定。第四批 5 条口径分歧没有因继续审核或时间经过自动解决。

累计开发清单为 [v5：66 条](../configs/splits/answer_repair_nq_train_confirmed_dev_v5.json)，对应 [C1 子集](../artifacts/answer_repair/nq_train_dev_v1/confirmed/through_batch05_c1.jsonl) 和 [运行时子集](../artifacts/answer_repair/nq_train_dev_v1/confirmed/through_batch05_runtime.jsonl)，含 11 个 E 目标及 55 个正确对照。均来自 development，校准／预留交集为 0，冻结评分预检通过；没有修改任何 gold、评分规则、原 C1 内容或调用 GPU。来源中同时保留用户意见与代理分歧，不能把审核计数当作修复收益。

网页可导入 [第五批累计状态](../artifacts/answer_repair/nq_train_dev_v1/review/NQ_train_pending_92.chat_confirmations_batch05_20261019.json)，显示本页 58 条确认、25 条待定、9 条空白。包含前几批网页范围内的全部意见；导入前先备份其他浏览器草稿，原 92 条包及哈希不变。

尚未审核仅余 N86/N87/N88/N90/N92/N93/N95/N97/N100。

## 第四批人工意见：2026-10-19（历史）

本批用户对 38 条给出意见，其中明确判对 33 条：N1/N2/N4/N5/N9/N10/N11/N12/N14/N15/N16/N17/N20/N21/N23/N24/N25/N26/N27/N30/N32/N34/N35/N36/N37/N38/N39/N40/N42/N45/N46/N47/N50。记录为正确对照，未进一步推定用户确认了独立的 `initial_supported` 评分。所有来源记录和引用位置校验通过。

另 5 条原话及代理复核均保存在 [第四批原始意见与待澄清口径](../data/annotations/human/answer_repair_nq_train_dev_v1_batch04_20261019.confirmations.json) 和 [逐条审核记录](../data/annotations/human/answer_repair_nq_train_dev_v1_batch04_20261019.jsonl)：

| 编号 | 用户意见 | 复核与处理 |
|---|---|---|
| N13 | 未答出在雪地上 | 题干已有 in the snow；C1 与 gold 均给仰卧及手脚动作。补写雪地可使说明更独立完整，是否构成语义错误待口径确认 |
| N18 | 不全面，应答 goal／作用 | C1 回答核心定义，题干问 meaning；R1/R2 的 understand how the universe behaves 可补充目标。定义题是否必须包括目标待确认 |
| N22 | 不算对，答动作，应答名词 | 原话“不算对”完整保留，`initial_correct=false` 仅表示用户这一形式裁决。C1 实为带分词修饰的名词短语，已列 Irish cream、whiskey、stout；代理保留语义可接受意见，未按事实错纳入 E |
| N41 | 没说小写 | C1 实际含小写字形 π。可补 lowercase，但字形本身已表达大小写；是否要求显式修饰词待确认 |
| N43 | 要写拉贾斯坦邦焦特布尔市 | R1 支持 Jodhpur, Rajasthan；C1 已给正确城市。用户的城市＋邦表达偏好保留，地点粒度标准待确认 |

已向用户请求对齐这 5 条的评测口径。尚无该问题的回复时，保留逐项意见及代理分歧，完整主标签为 null、开发用途为 hold；`human_review.status=deferred` 表示尚未完成共同口径裁决，不否认用户已经提供意见，也不覆盖原先 agent 建议。N22 的用户形式否定与代理语义判断明确区分；其他 4 条没有仅因补充意见就自动推定二元正确性为 false。建议补全答案另列，并标明不是模型输出，不改写 gold。

累计 47 条开发定案输入已导出至 [清单 v4](../configs/splits/answer_repair_nq_train_confirmed_dev_v4.json)、[C1 子集](../artifacts/answer_repair/nq_train_dev_v1/confirmed/through_batch04_c1.jsonl) 与 [运行时子集](../artifacts/answer_repair/nq_train_dev_v1/confirmed/through_batch04_runtime.jsonl)，含 11 个 E 目标、36 个正确对照；5 条分歧不进入该子集。原 gold、冻结 EM/F1、C1 原字节内容保持不变，校准／预留评测交集为 0。较多正确对照支持开发阶段检查误修，但有选择的开发集仍不能证明总体净收益。

网页最新累计状态文件为 [第四批导入 JSON](../artifacts/answer_repair/nq_train_dev_v1/review/NQ_train_pending_92.chat_confirmations_batch04_20261019.json)：本页 39 条确认、22 条待定、31 条空白。包含之前各批网页范围内的意见；先备份其他本机草稿，再手动导入。原 92 条材料及其包哈希不变。

剩余未审为 N51/N53/N54/N55/N58/N60/N61/N64/N65/N66/N68/N69/N70/N71/N72/N73/N75/N76/N77/N80/N83/N85/N86/N87/N88/N90/N92/N93/N95/N97/N100。

## 第三批人工意见：2026-10-18（历史）

本批严格只记录 N63/N67/N78/N79/N82/N84/N89/N91/N94/N98/N99 共 11 条。原话及范围见 [第三批确认](../data/annotations/human/answer_repair_nq_train_dev_v1_batch03_20261018.confirmations.json)，逐条结论见 [第三批人工记录](../data/annotations/human/answer_repair_nq_train_dev_v1_batch03_20261018.jsonl)。

| 编号 | 记录结果 |
|---|---|
| N63、N67、N78、N79、N82、N89、N91、N98 | 用户同意原暂缓意见，保留已知事实及未定字段，不自动归类；N82 已知原答案未回答固体分类，但证据是否足够仍未定 |
| N84 | 用户判对；R1/R2 直接支持 Henrietta Maria，R3 亦列为官方命名说法。加入正确对照，参考的替代说法不覆盖原答案 |
| N94 | 用户判对；接受本题职责概述粒度，加入正确对照。该个案不建立管辖范围限定普遍可省略的规则 |
| N99 | 用户确认题目所需内容没有在检索证据中给出。R4 有 Back-to-Africa 名称，却未支持“始于 1950s”的关系。单独确认 `evidence_sufficient=false`；参考的时间／题意疑点仍在，C1 正确性和完整主标签继续暂缓，候选方向为 Insufficiency，不纳入 E 核心 |

本批新增 2 条正确对照，无新增 utilisation 目标；其余 9 条保持暂缓，其中 N99 的证据缺失判断已单独记录。累计为 11 个 E 目标、3 个正确对照、17 条暂缓、69 条未审。原来的 23 条争议已全部收到人工意见，结果是 3 个 E 目标、3 个正确对照和 17 条暂缓；这是审核处理进度，不是 E 模型效果。

已导出 [累计 14 条开发清单 v3](../configs/splits/answer_repair_nq_train_confirmed_dev_v3.json)、[冻结 C1 子集](../artifacts/answer_repair/nq_train_dev_v1/confirmed/through_batch03_c1.jsonl) 和 [纯运行时子集](../artifacts/answer_repair/nq_train_dev_v1/confirmed/through_batch03_runtime.jsonl)，校准／预留交集均为 0。N52/N84/N94 仍保留原 EM=0，人审语义正确性另计；3 个有选择的正确对照仍不足以可靠估计总体误修率。N99 不进入这份可修复子集，没有重写 gold、补入外部事实或运行 GPU。

网页最新状态文件为 [第三批累计导入 JSON](../artifacts/answer_repair/nq_train_dev_v1/review/NQ_train_pending_92.chat_confirmations_batch03_20261018.json)：本页 6 条确认、17 条暂缓、69 条空白。原 92 条包及其哈希不变。先导出本机其他审核备份，再导入该文件；它包含第二、三批意见，不读取浏览器草稿，也不自动覆盖浏览器状态。

## 第二批人工意见：2026-10-18（历史）

本批只记录用户明确提及的 12 条。完整原话及逐条裁决见 [第二批确认](../data/annotations/human/answer_repair_nq_train_dev_v1_batch02_20261018.confirmations.json)，逐条状态见 [第二批人工记录](../data/annotations/human/answer_repair_nq_train_dev_v1_batch02_20261018.jsonl)。同意暂缓表示接受暂缓处理，未把正确性或主标签从 null 自动补齐。

| 编号 | 用户裁决与落实 |
|---|---|
| N3、N7 | 同意暂缓：题意与新旧名称／购买者和购买方式不明确 |
| N8 | 主体识别错误；本例按歌曲理解，R2 为 Deep Blue Something，C1 的 Audrey Hepburn 对应电影中的 Moon River；内部 E 映射为 utilisation |
| N19、N28 | 同意暂缓：证据关系与题意／参考仍有问题 |
| N33 | R1 清楚解释名称来源；C1 的地点事实有支持，但遗漏掉字招牌和三位成员加 3 的由来。保留 `partial`，内部 E 映射为 utilisation |
| N44 | 同意暂缓：相对时间未给起算点 |
| N52 | 用户确认答对；接受 September 2017 的本例时间粒度，作为正确对照。原文说宣布参赛月份，不据此扩展为所有宣布时间都等于实际事件时间 |
| N56、N57、N59 | 同意暂缓：媒介、计量维度及 ATP 计算口径等仍未决 |
| N62 | 本例应答全名 Lars Dittmann Mikkelsen；缺失主要姓氏 Mikkelsen，记录 `partial`，内部 E 映射为 utilisation |

本批新增 3 个 E 目标和 1 个正确对照，另 8 条明确暂缓。针对 N8/N33/N62 的 E 分类是依据用户对具体错误的确认和冻结证据作出的内部映射，未冒充上游 Diagnosis 正式标签。四条定案记录的 `initial_supported` 保留 null，用户没有单独确认这一评分；N33 的地点事实有支持与完整回答不足分别保留。N62 的裁决仅适用于本例缺少关键姓氏，不能推导所有人名都必须输出完整法定姓名。

已导出 [累计 12 条开发清单 v2](../configs/splits/answer_repair_nq_train_confirmed_dev_v2.json)、[冻结 C1 子集](../artifacts/answer_repair/nq_train_dev_v1/confirmed/through_batch02_c1.jsonl) 和 [纯运行时子集](../artifacts/answer_repair/nq_train_dev_v1/confirmed/through_batch02_runtime.jsonl)。均来自 development，与校准／预留评测交集为 0，runner 的冻结评分预检通过。N52 的原 EM=0 和 gold 保留，人审语义正确性另计。只有 1 条正确对照仍不足以可靠估计误修率；没有新 GPU 结果，也没有算法提升结论。

原网页保留同一 92 条材料和数据包哈希，避免使浏览器已有导出失效。更新页面校验以允许“部分正确但遗漏关键内容”的逐条人工 utilisation 裁决；不会自动把部分正确归为 utilisation。网页当前状态通过 [本次意见的导入 JSON](../artifacts/answer_repair/nq_train_dev_v1/review/NQ_train_pending_92.chat_confirmations_20261018.json) 同步：4 条已确认、8 条暂缓、80 条空白。需先重新打开／刷新更新后的 HTML，再点击“导入已有审核”。导入会替换该页面本机草稿；若浏览器还有其他审核，先导出备份。本文件只反映这次对话意见，不读取或覆盖任何浏览器草稿。

## 辅助初审分组

| 原筛查组 | 建议语义可接受 | utilisation 候选 | 暂缓 | 合计 |
|---|---:|---:|---:|---:|
| EM=0 且参考 passage 命中 | 49 | 8 | 23 | 80 |
| EM=1 对照 | 20 | 0 | 0 | 20 |
| 合计 | 69 | 8 | 23 | 100 |

上表保留初次 **agent 建议** 的分组。第一批 8 个 utilisation 候选得到用户确认后，原来剩余 92 条进入网页审核；第二批又对其中 12 条作出上述意见。逐条辅助理由依据全部 C1 Top-5 和离线参考，未查询外部事实，未使用 C2/C3 或新的 E 生成结果。引用均保留 passage_id、原文字符位置与证据哈希，100 条都在内部 development 部分。main 原始标签和冻结 scorer 均未覆盖。

`initial_correct` 表示是否回答了所求事实；`initial_supported` 表示证据能否支持它作为本题回答，不是文字是否出现。歧义案例中也会记录某一种解释有支持并保留正确性为 null，具体解释见理由。不能单独把这个辅助字段当作对脱离问题的陈述所做的 faithfulness 评分。

筛查组中的 49 条建议可接受，常见原因是有效别名、年份与完整日期、城市与国家附加信息、支持充分的改写、数字与英文数词、PTB 与 Unicode 标点。另一些涉及回答完整性或事实限定，需要人工裁决。69 中包括 20 条原有 EM 匹配对照；不能把 20→69 报成 E 的提升，E 在本批尚未运行。

本批刻意按 80:20 筛选，且只来自开发部分，不能把 8/80 或 69/100 外推为 NQ Train 的类别比例、语义正确率或总体收益。它支持的开发判断是：**EM=0 且命中 gold passage 只能提供候选，不能自动定义 utilisation。**

## 第一批已确认的 8 个目标错误（历史）

下列答案方向是代理阅读现有证据后的建议，不是 E 已生成的修复结果。N 编号属于 `answer_repair_nq_train_dev_v1`，与旧 v7 审核编号分开。

| 编号 | 问题与 C1 | 证据给出的答案方向 | 建议失败点 |
|---|---|---|---|
| N6 | Harry and the Hendersons 的 Harry；Graham Greene | Kevin Peter Hall | 采用 Twilight 同名角色的演员；R1 清楚支持目标作品 |
| N29 | Han 何时死亡；Han | Tokyo Drift | 人物名没有回答事件所在作品 |
| N31 | The Most Beautiful Girl in the World 歌手；Charlie Rich | Prince | 混淆近似但不同的歌名 |
| N48 | 印度最大人工湖；Dhebar Lake | Govind Ballabh Pant Sagar | 忽略 second-largest / after 的比较关系 |
| N49 | 亲吻青蛙使其变成王子的是谁；Princess Emma | 现代 Frog Prince 故事中的公主 | 所引改编版本中 Emma 自己变青蛙，转变结果不同；原代理置信度中等，用户已确认该判断 |
| N74 | Matrix Reloaded 的 Twins 是谁；fictional characters | Merovingian 的手下 | 类别过于笼统，没有回答具体身份 |
| N81 | True Blood 第七季集数；16 | 10 | 取了其他剧集第七季计数 |
| N96 | monkey bread 名称来源；Hungarian literature | 像猴子一样用手掰开食用 | 把菜肴传播史当作名称由来 |

用户已确认以上 8 条原答案错误、现有证据充分且清楚，按 utilisation 记录在 E 的人类审核层。原始 main 交付的 label 仍为 null，未冒充全量 Diagnosis 标注完成；表中的答案方向仍是人工/代理阅读后的建议，不是模型修复成功记录。

已固定 [8 条确认开发清单](../configs/splits/answer_repair_nq_train_confirmed_dev_v1.json)，并导出 [C1 输入](../artifacts/answer_repair/nq_train_dev_v1/confirmed/batch01_c1.jsonl) 和 [纯运行时输入](../artifacts/answer_repair/nq_train_dev_v1/confirmed/batch01_runtime.jsonl)。逐条匹配原始冻结 C1，仅从 development 选取，校准和评测预留重合均为 0，现有 E runner 接口校验通过。该批只有错误，没有已确认正确对照，不能用于估计总体净收益或误修率。

## 初次辅助审核的 23 条暂缓建议（历史）

| 编号 | 暂缓原因 |
|---|---|
| N3 | 问题残缺，Food Stamp Program 与 SNAP 的旧名/新名意图不明确 |
| N7 | purchased by 可问购买者，也可问购买方式 |
| N8 | Breakfast at Tiffany's 歌曲与电影中的 Moon River 歧义 |
| N19 | Gran Torino 的作曲/演唱关系及 Kyle/Clint 指代不清，现有段落不足以独立确认参考 |
| N28 | 题目问真实人物原型，参考给小说主人公，证据未说明原型 |
| N33 | 3 Doors Down 的取名地点与名称由来两种解读 |
| N44 | 相对升温时间缺少起算日期，历史暖夜纪录不能回答未来起点 |
| N52 | 宣布参赛月份与实际参赛时间/季数的对应需确认 |
| N56 | Boruto 漫画与动画发行时间不同，题目未指定媒介 |
| N57 | bigger 未指定半径、质量或体积，9 与 95 对应不同维度 |
| N59 | ATP 阶段产量与总产量、不同计量惯例冲突；参考存在范围问题 |
| N62 | 人名漏掉 Mikkelsen，不能只因是全名子串就视为有效别名 |
| N63 | 题干 President 与参考 Prime Minister 职务错位；缺少当年总统的证据 |
| N67 | 残缺题目可能问定义或学科归属 |
| N78 | Dinah 与 Cheshire Cat 都在相关作品中，题干没有限定哪只猫 |
| N79 | Scrabble 整套与手中数量、语言版本均未指定 |
| N82 | 回答用途而非固体分类；Top-5 对 molecular solid 的明确支持及参考表述需确认 |
| N84 | Maryland 名称有官方与替代解释，C1 有直接支持，参考不是唯一可接受来源 |
| N89 | over US$35 与 US$35 的数值容差及问题所指价格口径需确认 |
| N91 | closed system 是否排除外力矩、内部交换与总守恒的条件完整性需确认 |
| N94 | 删除 Commonwealth Agencies 管辖范围后是否仍为可接受概述 |
| N98 | 开始制作试播集与正式首播的事件歧义 |
| N99 | 题目 1950s 时间条件与参考早期思想来源不对应，且并非单一运动 |

暂缓不是新增主标签，也不证明 C1 一定错误。这些记录可继续搁置，不阻塞使用已确认的旧开发样本进行工程修改。N9/N18 的参考引用标记，以及 N59/N63/N84/N99 的参考疑点，均只记录问题，不改写 gold。

## 对 E 下一轮开发的约束

下一轮实际策略修改仍使用 v8 名称，本轮没有生成无修改的 v8 实验包。可以先以已有人类确认开发集推进工程，再逐步纳入本批已确认案例。

1. 先核验原答案是否已回答所求事实，支持充分时保留，减少无必要重答。核验只能看问题、原答案和运行时证据。
2. 需要修复时检查候选与问题的作品、人物、关系、时间、计量维度、比较及限定词是否匹配；仅逐字找到引用不够。
3. 同时保留引用原文定位，避免改成不可追溯答案；why/how 等问题按所求事实确定足够信息，不能统一机械裁成一个名称。
4. 将“原答案先核验”和“候选再核验”分别消融，与 C1、已有 v7 在同批开发输入上比较。记录改对、误修、无变化、解析失败与额外调用成本。
5. 冻结 EM/F1 继续单独报告，另报告经用户确认的语义正确性和证据支持。不能修改 gold 或放宽 scorer 来制造正收益。最终共同测试仍需与 Diagnosis 负责人协调，禁止用预留评测部分选规则。

这些是待验证的开发约束，并不表示某个核验模型或策略已有效。先用小批检查行为变化和误修，再决定是否扩大 GPU 实验。

## 保存与校验

- [92 条待审网页](../artifacts/answer_repair/nq_train_dev_v1/review/NQ_train_pending_92.html)：复用旧审核页的布局，可离线打开。23 条争议排在前面，随后是 69 条建议语义可接受；保留原 N 编号，排除此前确认的 8 条。
- [92 条待审完整原文](../data/annotations/assisted/answer_repair_nq_train_dev_v1_20261017.pending_92.full_evidence.txt)：与网页同一范围，包含原题、C1、参考、辅助理由和全部 460 段证据。
- [100 条可读判断](../data/annotations/assisted/answer_repair_nq_train_dev_v1_20261017.txt)
- [100 条原题、C1、冻结参考与完整 Top-5](../data/annotations/assisted/answer_repair_nq_train_dev_v1_20261017.full_evidence.txt)，先列 8 条目标候选，再列 23 条争议，N 编号不变
- [100 条结构化判断](../data/annotations/assisted/answer_repair_nq_train_dev_v1_20261017.jsonl)
- [计数、优先编号与来源校验](../data/annotations/assisted/answer_repair_nq_train_dev_v1_20261017.summary.json)
- [原始冻结审核材料与接入方法](ANSWER_REPAIR_NQ_TRAIN.md)

初次辅助审核的完整性检查通过：100 个 ID 与冻结清单顺序一致、均在 development；所有原记录哈希和逐字引用位置一致。原始审核包、代理判断、原文展示文件保留当时 pending 的冻结快照；当前状态以累计人类审核层为准。第一批确认文件严格保留上述 8 个 ID，第二批另存，原始 92 条网页材料范围保持不变。main 交付文件、源分片、gold、冻结 scorer 和 v7 结果未改。辅助记录 SHA-256：`920ea1c56f1b9512346e0d84fcd1d8e9e8e6abe9e62478e7ba2392378dece389`。

## 网页审核方法

用浏览器打开 `NQ_train_pending_92.html`，无需启动服务或联网。左侧可按原 N 编号跳转、搜索或筛选；每条展示原题、C1、可展开的冻结参考和完整 Top-5，右侧填写判断，辅助理由可单独展开。

人工字段初始留空。对 69 条建议可接受的样本，点击“采用辅助建议为草稿”只填入草稿；核对后仍需填写审核人并点击“确认本条人工审核”。争议无法裁定时填写原因并“保存为暂缓”；暂缓不会计入已确认数。修改已确认记录会使它恢复草稿。

草稿保存在当前浏览器本机，浏览器限制或清理数据可能使其丢失。每审核一批就点击顶部“导出审核 JSON”备份；可在同一版本网页导入后续审。导入会校验审核包、样本 ID、原 N 编号和来源哈希，再提示替换已有草稿。清空只影响本页本机草稿，源文件和此前 8 条确认不变。

导出文件交回后再按逐条明确的人工操作核对入库。网页不会自动修改仓库；旧 v6/v7 审核导入器不适用于本页的 `nq_train_e_human_review_v1` 格式。页面、辅助建议和界面测试均不增加任何人工确认。

可复现构建：`.venv/bin/python scripts/build_nq_train_pending_review.py`。构建器先校验冻结 100 条、辅助判断和首批 8 条确认的来源及 92 条原始范围，再生成 HTML 与同名 manifest；后者记录 HTML、输入和构建源码的 SHA-256。UI 更改不改变审核数据包哈希，便于继续导入已有草稿。针对范围完整性、原文保留、导入身份校验、确认条件和草稿状态的 3 项 Python 检查及 6 项 Node 检查通过。

预览中已核对显示范围、证据展开、跳转、筛选、草稿持久化、导出和同包导入。最后一轮预览的点击操作未响应，因此新增的页面内清空／覆盖确认弹窗未完成浏览器复核；不把这一项记为通过。界面测试仅产生未确认草稿，没有导入仓库标注。
