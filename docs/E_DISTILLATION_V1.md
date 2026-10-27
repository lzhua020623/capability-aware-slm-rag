# E 输出蒸馏准备 · 2026-10-27

分支 `controller_2`。本轮启动新的 NQ 蒸馏扩展，已准备数据、契约和离线教师任务，并在 Azure 建好教师部署及受控试产入口；不覆盖旧 r2 失败结果，也不改变当前 E unavailable、C1 权重、检索资源或路由。[首次 3 条真实连接检查](../artifacts/controller/e_teacher_sol_connection_v1/README.md)已通过版本、来源和结构重放，完整试产、语义验收、SFT 和 GPU 验收尚未完成。

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
- 教师已选 Azure GPT-6.1 Sol `2026-09-29`、Global Standard 按量计费、100K TPM。部署 `elec5623-e-teacher-sol-v1` 的状态已核对为 Succeeded；原 ELEC5623 项目区域不支持该模型，因此使用用户同一订阅下已有的兼容项目，没有另建资源。密钥不提交到仓库。

## 教师调用登记

独立 [执行登记](../configs/distillation/e_teacher_sol_pilot_v1.json)引用原准备协议与完整任务指纹，原准备协议保持字节不变。选择 Sol 的原因是它提供结构化输出及推理能力，当前 Azure 公布的普通短输入单价为每百万输入 $2、输出 $10，低于 Astra 的 $10/$50；这不证明它在本任务上足够好，仍需试产验收。[Microsoft 价格表](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/provisioned-throughput-sizing)

- **版本限制**：Azure 当前策略仍为 `OnceNewDefaultVersionAvailable`，页面未能关闭自动升级。客户端要求每个 response.model 精确为 `gpt-6.1-sol-2026-09-29`；不匹配即停止，该条不接受。不能把部署本身写成禁止自动升级，也不能声称调用前已通过管理 API 实时验证版本。
- **预算**：这一份试产账本最多 500 次请求、15 USD 的客户端费用上限。请求先预留完整输入包 UTF-8 字节数加 4096 的 token 上界，以及 4096 个总输出 token；拒绝输入上界超过 90K 的请求。按每百万输入 $5／输出 $15 保守记账，覆盖当前公布的输入缓存写入及长上下文单价。成功后按 usage 结算，reasoning 已在 completion_tokens 中，不重复计算。这是当前登记价格下的上界估计，不能当 Azure 账单，也不控制其他应用花费或未来价格变化。
- **失败**：超时、用量缺失、版本不符、拒绝、截断、非法引用或契约不合规都立即停止，保留完整预留费用；不自动重试，不覆盖原响应。进程中断时已预留的请求同样阻止重发。
- **默认运行**：无 `--execute` 仅验证。实际调用默认累计 3 条，继续运行使用同一固定目录和账本跳过已成功 ID；显式指定累计 500 才扩到完整试产。四种冻结 AI 分层轮流调用，预算停止不会只留下第一类。为低于部署限流，初版单进程最多每分钟 1 次，500 次至少约 8.5 小时；这不消耗本地 GPU。
- **教师和学生预算分开**：教师 4096 含内部推理及可见输出，medium reasoning；只保存 provider 返回的可见 JSON、用量和请求信息，不请求隐藏思维链。学生目标仍须逐条验证不超过 Qwen 的 384 tokens。
- **质量门槛**：扩量前全量来源/契约合规率至少 95%；按固定哈希抽审至多 50 条替换（至少需有 25 条），剩余名额在支持并保留、缺证并保留、明确错误但无候选三类间轮流补至共 100 条；稀少保留类全部纳入后再分配余量。三种核验标签各至少 10 条。语义正确至少 95 条，审核替换中不能有误修，待定按未通过；少于 100 个合规输出或修复/标签覆盖不足则不扩量。格式检查不会批准 SFT，所有入训目标另做 384-token 检查。该门槛是本次试产的工程决策，不能据此宣称总体误修率为零。

```bash
# 只做本地复核，不调用 Azure
python scripts/run_e_teacher_pilot.py

# 在本机终端配置实际 Azure v1 base endpoint；API key 不放命令行/聊天
export AZURE_OPENAI_ENDPOINT='https://<已登记的资源>.services.ai.azure.com/openai/v1/'
python scripts/run_e_teacher_pilot.py --execute --prompt-key
# 同一账本下的后续完整试产（先核查前三条，不会重复已成功请求）
python scripts/run_e_teacher_pilot.py --execute --prompt-key --total-requests 500
```

endpoint 必须与实际观察的资源哈希一致；其他资源、重定向、含 key 的 URL 均被拒绝。`--prompt-key` 用终端隐藏输入读取已有密钥，仅存本次进程内存；也可预先配置 `AZURE_OPENAI_API_KEY`。用户已通过隐藏输入完成首次 3 次真实调用，密钥未交给代理或写入仓库。不要发送密钥给对话。

首次检查实测：两次 supported、一次 insufficient，均保留原答，Qwen 目标含 EOS 分别 86、66、70 tokens；保守客户端费用合计 $0.04104，实际 Azure 账单未核实。它只验证连接与输出契约，没有覆盖真实替换，也不是教师语义通过。Texans 一题引用的是 Texans/Chiefs 旧球队链条，与离线参考存在实体对齐风险，必须进入语义待审；不能因为格式通过就加入 SFT。之后从同一账本第 4 条继续，完整试产质量门槛不变。

## 节流补充 · transport v2

500 条是 Azure 远端教师调用，本机 CPU 只发送请求，不运行教师模型。换到 Kaggle GPU 不会消除客户端等待；GPU 用于后续学生推理和 QLoRA。原每 61 秒一条是初版保守策略，已观察到单条服务时间约 2.5–7.3 秒后，新增独立 [transport v2 登记](../configs/distillation/e_teacher_transport_v2.json)，原教师登记、请求体、样本顺序、预算及质量门槛均保持原字节。

v2 需显式启用，串行最多一条在途，任意滚动 61 秒完整 token 预留总额不超过 95K、请求数不超过 60，开始时间至少相隔 5 秒。按完整请求 UTF-8 字节上界＋4096 输入 allowance＋4096 最大输出预留；不按少量实际输出提前退还 rate 预留。全部 500 条预估范围 15,705–22,922 tokens/条，均通过窗口检查。[Microsoft 文档](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/quota)说明 rate 估算包含输入和最大输出，与计费用量不同。

```bash
python scripts/run_e_teacher_pilot.py --execute --prompt-key \
  --total-requests 500 --transport-profile token_window_v2
```

继续用同一个目录、进程锁和累计费用账本；已完成样本不重发。恢复时先等待一个完整窗口，清除旧进程未知的 rate 时间，再开始发送。新 profile 和客户端/调度器源码哈希在发送前追加到账本。原失败停止、不自动重试和不保存凭据保持不变。Azure 临时限流仍可触发停止，不能把 95K 本地估算写成服务保证。

终端现在打印 resume、rate_wait 和 completed 进度，只含累计条数、标签/动作和保守费用，不打印问题、证据或凭据。正在运行的旧 Python 进程不会热更新；切换要在无未结算请求时停止旧进程，再用同一账本启动 v2、重新隐藏输入密钥。不能并行起两个 writer，不能强杀未结算请求后自动重发。

离线模拟剩余 490 条：单条服务时间设为 5／15／30 秒时，v2 分别约 1.71／2.06／4.10 小时；旧策略约 8.30 小时。这是当前输入形状的调度模拟，不是 GPU 或 Azure 端到端实测，不承诺固定完成时间。真实连接和教师语义结果继续按原协议记录。

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

1. 教师部署、执行预算、来源守卫及 API transport 已登记并实现；首次 3 条连接/契约检查已完成并独立重放，不静默换模型、放宽格式或另开账本。
2. 沿用同一账本产生至多 500 条教师输出并执行已登记质量审核。某类不足、全部保留或全部 insufficient 都不能通过扩量；不能拿被审核后的结果修改本轮门槛。
3. 教师能力和来源检查通过后，再扩量并构建 SFT 数据；tokenizer 检查所有目标长度，截断不得移除依据后保留原监督。
4. 实测训练环境并冻结至多两个训练配置、工程 fixtures、正确题误修上限、置信区间与晋升规则。任何训练前这些字段都必须具体化；本准备协议不冒充最终执行封存。
5. 固定 C1、Top-5、提示、解码和预算，比较保留 C1、原 E、蒸馏 E。开发选择一个候选，Selection 确认一次；同时报告语义修复/误修、EM/F1、覆盖与成本。
6. 达标后接入固定路由；P60 和共同 baseline 未齐时不宣布集成收益。FEVER 未统一前不进入本轮。DPO、RSI 和知识库修改另列，不同时改变。

教师 API 准备依据 [OpenAI Docs Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)：传输格式约束不证明语义正确，拒绝/截断仍需单独处理。训练实现参考 [TRL SFT](https://huggingface.co/docs/trl/main/en/sft_trainer) 和 [PEFT 量化/QLoRA](https://huggingface.co/docs/peft/main/en/developer_guides/quantization)；实际环境须锁定版本后验收。
