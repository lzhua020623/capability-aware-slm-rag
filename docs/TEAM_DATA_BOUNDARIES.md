# 团队数据对齐与用途边界 · 2026-10-27

当前分支 `controller_2`。本轮将团队原始划分按原字节引入，并收紧后续数据准备、训练和当前生产入口的用途检查。没有重划分、修改测试 ID、启动训练或教师 API。

## 团队的唯一 NQ 划分

来源：`feature/baseline-evaluation@387ddafc82bc939ebd27b3137d580db6ecb2e04a`。

- 文件：[diagnosis_nq_train_74097_split.json](../baselines/splits/diagnosis_nq_train_74097_split.json)
- SHA-256：`7da01c94cd7b5f92076edebc80911ac891d3861a01cf6b40a1e5e6efb6482dc4`
- 登记：[team_nq_data_v1.json](../configs/splits/team_nq_data_v1.json)，由当前生产配置绑定其哈希。

| 用途 | 团队原始数量 | 本轮保守允许的新实验池 |
|---|---:|---:|
| Train | 51,866 | 46,421 |
| Validation | 11,117 | 9,440 |
| Test | 11,114 | 原 ID、顺序及数量全部保留；正式执行须另立 T7 封存 |

额外排除是用途限制，不是重新划分：Train 仍排除旧完整保留池和旧 Controller selection；Validation 进一步排除已登记的下游开发/分析曝光。不会把被排除的题转给另一个角色。

## 已实际核查的内容

- 本地标注索引与团队划分均为 74,097 个唯一 ID，缺失、多出、重复均为 0；团队三组 ID 无交集。
- 本地初标按团队 Train/Validation 聚合的标签数量，与划分文件中的对应元数据完全一致；这不能证明每个语义标签正确，也不能替代原始 Diagnosis 文件的逐 ID 标签比对。
- 15 个原始 C1 分片都在 `data/processed/external/nq_train_delivery_20261017/`，逐份 SHA-256 与入库交付 manifest 一致。原始数据集没有缺失，不需要重新下载。
- 对允许使用的 46,421 条 Train 和 9,440 条 Validation 流式检查：query、原答、Top-5 数量/顺序/唯一 ID/原文/有限分数、gold 参考字段均无结构异常；规范化 query 指纹跨这两组的重复为 0。
- Test 的问题、证据和 gold 不进行解码检查。未以读取 test 内容的方式重做近重复筛查；不把团队原分组声明当成新的独立内容审计。

独立复查工具：

```bash
python scripts/audit_team_nq_data.py --output-dir /tmp/team-nq-audit-new
python scripts/audit_team_nq_data.py --ids-only --output-dir /tmp/team-nq-ids-new
```

输出目录必须全新。完整检查只解码允许角色的 C1 内容；test/受保护记录先按 ID 跳过。工具不改数据或模型。

## 历史曝光扩大核对

[known_exposure_ids.json](../artifacts/controller/team_data_v1/known_exposure_ids.json) 按已保存的 E、Controller 和重排实验 ID 重算，记录文件哈希和来源。

- 旧 E 登记、Controller 开发/选型的 130 条只是前一次核对范围。
- 加入重排目标类、正确类及 pilot 等已保存结果后，共有 **603 条不同的团队 test ID** 已进入下游实验或分析。它们不是“603 条答错”，也不等于“603 条已训练”。
- 其余 **10,511 条**仅表示“不在当前登记的曝光来源中”，不是团队完全未见的证明。旧训练权重的实际训练/调参 ID 尚未全部交付，潜在训练重叠仍未排除。
- 最终主表继续对齐团队 11,114 条，披露已知曝光，并补报排除已知曝光的配对分析；不偷偷替换成另一个测试集。

## 可执行边界

实现：[src/data/team_nq.py](../src/data/team_nq.py)。

1. 固定原始 Git 提交与 split 指纹；即使重新分配后保持相同数量、再重算配置哈希，也拒绝冒充原团队划分。
2. ID 必须是精确字符串；未知 ID、浮点/整数强转、跨角色、旧保留池、已曝光 Validation 均拒绝。独立外部数据需要单独来源登记。
3. 先按团队角色分开，再生成训练正负例；答案选择器的借用答案只来自同一角色，并记录 donor_sample_id。合成/增强样本需要可核查的 parent ID。
4. 两个 builder 不再在 74,097 条上另做随机 70/15/15 或 90/10。新数据输出仅含 train/validation，没有 test.jsonl，且不覆盖旧目录。
5. 两个 trainer 在任何模型加载、CUDA 设置及样本截取之前，验证策略源文件指纹、完整训练文件哈希、角色、donor/parent 及规范化问题交集。builder 另记录原始输入文件指纹。旧无回执数据必须重新准备。
6. 重排 trainer 不读取或自动评分本地 test；重排比较和端到端组件评测默认使用登记后的 Validation。自定义子集和旧参数不能绕过用途检查。
7. 当前生产入口默认 data_role=validation，逐题在 Diagnosis/增强前检查角色并记录策略/划分指纹。普通入口不能请求 test；FEVER 没有统一登记前也不能套用 NQ 的角色。低层虚构测试注入会明确标记未执行生产数据边界。
8. 未来教师输入通过 teacher_input() 仅保留 sample_id、dataset、query、initial_answer、evidence；gold、AI 标签及额外证据标注不进入请求。教师数据生成限允许的 Train。

旧冻结包仍可按各自历史协议复现，不改写历史配置和结果；上述是当前分支的新实验入口。P60 若实际使用独立 NQ Dev 训练，需单独登记来源和跨集隔离，不能伪装为本次 Train 数据。

## 数据体检不等于语义验收

这些 Diagnosis 标签仍是 AI 初标。原答与 gold 的语义等价、限定词、弱监督负例、答案选择器的 gold 正例是否被证据支持，都未因结构检查通过而成为人工 gold。新生成数据回执保留 labels_semantically_verified=false；不能直接当作高质量 E 蒸馏目标。

历史 74,097 条 C1 和新团队 baseline 的 context_fallback 输出也不是同一份预测。前者可以用于登记后的开发/训练；最终配对比较仍需后者的 predictions、metrics、run_config。当前缺的是新 baseline 逐题输出，不是原始样本或检索库。

检索语料库和索引保持与 baseline 匹配。用途隔离针对 QA 样本及监督字段，不从维基知识库中删除测试题相关的正常检索段落。

FEVER 的团队划分、P60 的原始在线特征和模型使用 ID、候选重排/选择器的训练来源、新 baseline 逐题交付仍是后续待办。E 仍 unavailable，未因本轮体检自动启用训练权重。
