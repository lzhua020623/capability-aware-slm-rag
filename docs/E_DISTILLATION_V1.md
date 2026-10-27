# E 输出蒸馏准备 · 2026-10-27

分支 `controller_2`。本轮启动新的 NQ 蒸馏扩展，只准备数据、契约和离线教师任务；不覆盖旧 r2 失败结果，也不改变当前 E unavailable、C1 权重、检索资源或路由。真实教师输出、SFT 和 GPU 验收尚未开始。

## 组件复查结论

本轮刷新主仓库全部远端分支，并核对两个已知 fork，无新增提交。Diagnosis `138d52a`、补检索 `37e469e`、重排 `5583285`、E `5da1fe5` 均已在当前历史中；集成层保留后续兼容修正，不要求与分支原文件逐字节相同。

上段记录首轮检查。准备提交后再次刷新，重排新增三次提交至 `1d51856`，已做 [后续兼容集成](RERANK_1D51856_INTEGRATION.md)：补上新 selector 评测的用途边界、深候选池资源登记和 frozen 后端参数检查。当前 E 契约及样本未变。

最新源码已合入，不等于全部组件可以生产运行：P60 缺原始 78 维在线特征和三个 HGB 的使用 ID，启动检查退出 2；新重排权重和答案选择器未晋升；E r2 未通过真实验收。团队 baseline 分支没有整体合并，原始 NQ split 已按指纹独立接入，其新 context_fallback 逐题输出仍待交付。

因此可以独立准备 E 训练数据，但不能启动新的端到端效果评测。组件审计与运行回执在 [本轮归档](../artifacts/controller/e_distillation_v1/README.md)。

## 本轮固定的内容

- [准备协议](../configs/distillation/e_prep_v1.json)绑定现有 v2a 的提示、解析器和数据边界源码哈希。
- 学生为现有 `Qwen/Qwen2.5-7B-Instruct`，沿用仓库登记的 revision。候选方法为 QLoRA/SFT，只训练 E 的 adapter；C1 使用原权重。训练库版本、显存、adapter 参数和长输入方案需在实际训练前登记实测。
- 输入仍只有 sample_id、dataset、query、initial_answer、Top-5。教师没有 gold、AI 标签、Diagnosis 结果、工具或外部检索。
- 模型输出沿用 original_verification、verdict_evidence_ids、replacement_answer、replacement_evidence_ids、reason 五字段。supported 保留；contradicted 仅在有合格候选时替换；insufficient 保留并标记。最终动作由程序执行，不另训一套矛盾的动作字段。
- 教师给出至多 60 词的简短可核查理由。学生输出预算仍为 384 tokens；词数限制不能代替真实 tokenizer 检查。教师的可见输出与内部推理总预算分别登记，不默认把 384 当作教师总预算。
- 教师候选平台为 Azure，实际模型、版本、部署、endpoint 和预算尚未登记。密钥只放环境变量，不提交到仓库。当前工具没有网络发送能力。

## 样本安排

团队划分和旧保留池限制不变，使用 [NQ 用途边界](TEAM_DATA_BOUNDARIES.md)。

| 清单 | 数量 | 取样与用途 |
|---|---:|---|
| Teacher pilot | 500 | 从允许的 46,421 Train 中，对四种 AI 初标各取 125；只用于教师质量/成本试产 |
| Development | 500 | 从允许的 9,440 Validation 按固定哈希均匀抽取，开发评测 |
| Selection | 500 | 从同一允许 Validation 池抽取，与 Development 不重叠；候选确认一次 |
| Final test | 11,114 | 保持封存，当前工具不能读取其问题/证据/gold |

取样只读 ID 和 AI 元数据，不根据 gold、答案或试跑得分筛题。Pilot 是有意平衡的诊断样本，不代表自然分布，更不是 500 条已确认 utilisation。开发/选型仍是团队 validation 的用途子集，不重新划分团队数据。

通过教师试产后，候选目标是 5,000 条合格混合监督样本，上限 10,000；扩量清单、预算和质量门槛须另行封存。不能把本轮 500 条试产当成完整训练规模，也不能直接复制 AI 标签作为训练目标。

## 工具与输出

```bash
python scripts/prepare_e_distillation.py \
  --manifest artifacts/controller/e_distillation_v1/sample_manifest.json \
  --export-pilot data/processed/e_distillation_v1/pilot_20261027
```

重新验证清单时只给 `--manifest`；输出目录必须全新，既有试产不能覆盖。

- `teacher_tasks.jsonl`：五字段 runtime、原学生提示、教师消息、对象根输出 schema 和内容指纹。它是离线任务，不是可以直接提交的 API Batch 文件。
- `offline_references.jsonl`：物理分开的 train gold/AI 元数据，仅供离线筛查，不能随教师任务上传。
- `inputs_manifest.json`：原始 15 分片哈希、样本清单指纹、两个导出文件哈希和待办。完整任务/参考保存在被忽略的 data/processed；仓库只记录 ID 和审计摘要。

工具先按 ID 排除 Validation/Test/保留池，再解码 Train。教师提示生成前，离线字段由白名单剔除。现有学生 v2a schema 保留不变；教师传输 schema 使用 object 根，跨字段与来源约束在本地继续检查。

`audit_teacher_output()` 只认证格式和来源，永远返回 semantic_verified=false、approved_for_SFT=false。它可发现引用不存在、片段不在原文、错误改答分支和等价改答等情况；它不能证明问题关系、限定词或答案含义正确。

## 下一步和停止边界

1. 登记实际教师部署、模型快照、输出/推理 token 与金额上限、重试策略，确认 provider 的 schema 子集。之后才加入和执行 API transport。
2. 封存教师试产质量门槛与抽审办法；先产生 500 条教师输出，程序检查全量，按输出行为抽审并隔离分歧。缺少修复样例、全部保留或全部 insufficient 都不能算通过。
3. 教师能力和来源检查通过后，再扩量并构建 SFT 数据；tokenizer 检查所有目标长度，截断不得移除依据后保留原监督。
4. 实测训练环境并冻结至多两个训练配置、工程 fixtures、正确题误修上限、置信区间与晋升规则。任何训练前这些字段都必须具体化；本准备协议不冒充最终执行封存。
5. 固定 C1、Top-5、提示、解码和预算，比较保留 C1、原 E、蒸馏 E。开发选择一个候选，Selection 确认一次；同时报告语义修复/误修、EM/F1、覆盖与成本。
6. 达标后接入固定路由；P60 和共同 baseline 未齐时不宣布集成收益。FEVER 未统一前不进入本轮。DPO、RSI 和知识库修改另列，不同时改变。

教师 API 准备依据 [OpenAI Docs Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)：传输格式约束不证明语义正确，拒绝/截断仍需单独处理。训练实现参考 [TRL SFT](https://huggingface.co/docs/trl/main/en/sft_trainer) 和 [PEFT 量化/QLoRA](https://huggingface.co/docs/peft/main/en/developer_guides/quantization)；实际环境须锁定版本后验收。
