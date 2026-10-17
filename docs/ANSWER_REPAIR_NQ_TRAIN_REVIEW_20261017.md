# NQ Train 开发样本辅助审核：2026-10-17

已完成固定首批 100 条的代理辅助审核。随后用户在阅读 8 条目标候选的原题、C1、冻结参考和完整 Top-5 后回复“判断可以”，确认 N6/N29/N31/N48/N49/N74/N81/N96 的判断。当前 **8 条人工确认、92 条待审**；待审包括 69 条建议可接受和 23 条争议。原始数据以 `origin/main@058f10baa47083220f95edb39db8f106aa7e85d6` 为准；核对时未发现该提交之后的新 main 提交。

确认记录见 [用户原话与明确范围](../data/annotations/human/answer_repair_nq_train_dev_v1_batch01_20261017.confirmations.json)、[8 条人工记录](../data/annotations/human/answer_repair_nq_train_dev_v1_batch01_20261017.jsonl) 和 [当前计数](../data/annotations/human/answer_repair_nq_train_dev_v1_batch01_20261017.summary.json)。这是查看代理分析及完整证据后的对话确认，不是独立盲标；没有确认任何 E 新输出正确。

本轮核对 main 新增的 11 个交付/流水线文件与本地版本，均逐字节一致；Google 下载的 15 个原始分片均与 main 清单的 SHA-256 一致。74,097 条源记录及其 gold 未修改。E 的转换文件、内部划分与审核建议均为派生产物；如与 main 后续正式协议或团队共同划分冲突，应按上游重建新的派生版本。

## 辅助初审分组

| 原筛查组 | 建议语义可接受 | utilisation 候选 | 暂缓 | 合计 |
|---|---:|---:|---:|---:|
| EM=0 且参考 passage 命中 | 49 | 8 | 23 | 80 |
| EM=1 对照 | 20 | 0 | 0 | 20 |
| 合计 | 69 | 8 | 23 | 100 |

上表保留初次 **agent 建议** 的分组。其中 8 个 utilisation 候选现已得到用户确认，可用于 E 开发；其余 92 条不能用作已确认目标。逐条理由依据全部 C1 Top-5 和离线参考，未查询外部事实，未使用 C2/C3 或新的 E 生成结果。引用均保留 passage_id、原文字符位置与证据哈希，100 条都在内部 development 部分。main 原始标签和冻结 scorer 均未覆盖。

`initial_correct` 表示是否回答了所求事实；`initial_supported` 表示证据能否支持它作为本题回答，不是文字是否出现。歧义案例中也会记录某一种解释有支持并保留正确性为 null，具体解释见理由。不能单独把这个辅助字段当作对脱离问题的陈述所做的 faithfulness 评分。

筛查组中的 49 条建议可接受，常见原因是有效别名、年份与完整日期、城市与国家附加信息、支持充分的改写、数字与英文数词、PTB 与 Unicode 标点。另一些涉及回答完整性或事实限定，需要人工裁决。69 中包括 20 条原有 EM 匹配对照；不能把 20→69 报成 E 的提升，E 在本批尚未运行。

本批刻意按 80:20 筛选，且只来自开发部分，不能把 8/80 或 69/100 外推为 NQ Train 的类别比例、语义正确率或总体收益。它支持的开发判断是：**EM=0 且命中 gold passage 只能提供候选，不能自动定义 utilisation。**

## 已确认的 8 个目标错误

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

## 23 条暂缓

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

初次辅助审核的完整性检查通过：100 个 ID 与冻结清单顺序一致、均在 development；所有原记录哈希和逐字引用位置一致。原始审核包、代理判断、原文展示文件保留当时 pending 的冻结快照；当前状态以人类审核层为准。确认记录严格只包含上述 8 个 ID，其他 92 条未改变。main 交付文件、源分片、gold、冻结 scorer 和 v7 结果未改。辅助记录 SHA-256：`920ea1c56f1b9512346e0d84fcd1d8e9e8e6abe9e62478e7ba2392378dece389`。

## 网页审核方法

用浏览器打开 `NQ_train_pending_92.html`，无需启动服务或联网。左侧可按原 N 编号跳转、搜索或筛选；每条展示原题、C1、可展开的冻结参考和完整 Top-5，右侧填写判断，辅助理由可单独展开。

人工字段初始留空。对 69 条建议可接受的样本，点击“采用辅助建议为草稿”只填入草稿；核对后仍需填写审核人并点击“确认本条人工审核”。争议无法裁定时填写原因并“保存为暂缓”；暂缓不会计入已确认数。修改已确认记录会使它恢复草稿。

草稿保存在当前浏览器本机，浏览器限制或清理数据可能使其丢失。每审核一批就点击顶部“导出审核 JSON”备份；可在同一版本网页导入后续审。导入会校验审核包、样本 ID、原 N 编号和来源哈希，再提示替换已有草稿。清空只影响本页本机草稿，源文件和此前 8 条确认不变。

导出文件交回后再按逐条明确的人工操作核对入库。网页不会自动修改仓库；旧 v6/v7 审核导入器不适用于本页的 `nq_train_e_human_review_v1` 格式。页面、辅助建议和界面测试均不增加任何人工确认。

可复现构建：`.venv/bin/python scripts/build_nq_train_pending_review.py`。构建器先校验冻结 100 条、辅助判断和 8 条确认的来源及剩余范围，再生成 HTML 与同名 manifest；后者记录 HTML、输入和构建源码的 SHA-256。UI 更改不改变审核数据包哈希，便于继续导入已有草稿。针对范围完整性、原文保留、导入身份校验、确认条件和草稿状态的 3 项 Python 检查及 5 项 Node 检查通过。

预览中已核对显示范围、证据展开、跳转、筛选、草稿持久化、导出和同包导入。最后一轮预览的点击操作未响应，因此新增的页面内清空／覆盖确认弹窗未完成浏览器复核；不把这一项记为通过。界面测试仅产生未确认草稿，没有导入仓库标注。
