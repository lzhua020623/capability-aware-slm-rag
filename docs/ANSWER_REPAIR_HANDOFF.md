# E 交接清单（2026-10-14）

E 负责 Evidence Utilisation 的 Verification / Answer Repair：接受 query、C1 Top-5 和初始答案，按 Controller 的最终 generation_repair 决策运行，返回最终答案、引用、处理状态及成本。Diagnosis/Probe/Controller 阈值仍由上游负责人维护。

2026-10-14 晚更新：关键案例已完成第二次独立盲评（与辅助审核一致），确认清单见 `ANSWER_REPAIR_CONFIRMATION_20261014.md`；v7 已冻结打包但未运行，见本文末尾“v7 交接”。

当前状态：工程实现、本轮计划的真实 GPU 实验、自动验收、归档和文档交接已完成。v6r1 的 128 条输出已通过冻结代码回放；可靠算法收益尚未验收，不能把整体研究验收标成完成。人工审核按用户指令暂缓，所有新输出的人工字段保持 pending/null。交付分支为 `feature/verification-answer-repair`，实现、冻结实验归档、结果和辅助审核一并纳入版本控制；最新提交以该分支的 Git 记录为准。

| 内容 | 状态 | 入口或依据 |
|---|---|---|
| 统一修复接口、结构校验、证据来源与回退 | 已实现 | `src/compensation/answer_repair.py` |
| 原答案隐藏的候选、同模型二次核验 | 已实测并保留负结果 | `ANSWER_REPAIR_V5.md`，112 条真实输出 |
| NQ 只生成一次原文片段 | 已实测，未晋升默认 | dev50 EM 20→7，长引文与无关片段需人工复评 |
| FEVER 独立 MNLI、分窗和冲突回退 | 已实测，未晋升默认 | 审核后 4→7，但 dev50 45→44，存在实体误判 |
| 接受上游 ControllerDecision，其他 action 零调用 | 已验收兼容性 | `controller_adapter.py`，100 条历史真实决策回放 |
| 断点续跑、模型版本、两类成本、结果指纹 | 已检查 | `scripts/run_answer_repair.py` |
| 固定训练开发集及隔离 validation/test | 已冻结 | `configs/splits/answer_repair_reviewed_v6.json` |
| 单元与集成检查 | 81 项通过 | 当前代码与冻结包分别运行 |
| 真实 tokenizer 分窗完整性 | 64 输入通过 | 365 窗口，未执行模型推断 |
| v5 原始输出和冻结代码 CPU 回放 | 112 条通过 | `v5_reviewed/release_audit/cpu_replay.json` |
| v6r1 28 条 + 原 dev100 GPU 实验 | 已完成、128 条回放通过 | 123 个唯一 ID；两批分别报告 |
| 新输出前后答案成对人工验收 | 用户要求暂缓 | v5 112 条、v6r1 128 条待审核；新版 42 条改动优先 |
| 9 条初始边界样本 | 用户要求暂缓 | cohort manifest 的 deferred |
| 全框架 held-out 与三补偿操作比较 | 团队后续 | 不属于本次 E 单模块完成证明 |
| 关键案例独立盲评二读 | 已完成，待用户确认 | `ANSWER_REPAIR_CONFIRMATION_20261014.md`；9 修复 / 3 误修两读一致 |
| v7 策略、测试、离线回放、冻结包与 notebook | 已完成，未跑 GPU | `ANSWER_REPAIR_V7.md`；99 项测试；`artifacts/answer_repair/v7_design/`、`v7_reviewed/` |
| E 未触碰的 held-out（FEVER 267 / NQ 146） | 已冻结在 v7 包内 | 只跑一次；无 utilisation 标注，用于误修与 EM 假退步评估 |
| NQ 3216 池 held-out（含 120 条 utilisation） | 已冻结打包，未跑 GPU | gold 来自 Diagnosis 负责人工作簿；`artifacts/answer_repair/v7_heldout3216/`，只跑一次 |

## 运行和结果入口

Colab notebook：`notebooks/answer_repair_colab_v6r1_reviewed.ipynb`。
Kaggle notebook：`notebooks/answer_repair_kaggle_v6r1_reviewed.ipynb`，适配平台自动解压，固定校验清单的哈希后逐文件校验 48 个文件。模型、算法、阈值与样本保持同一冻结版本。
便携包：`artifacts/answer_repair/v6r1_reviewed/answer_repair_v6r1_reviewed_gpu.tar.gz`。
完整操作见 `ANSWER_REPAIR_COLAB.md`；运行前协议见 `ANSWER_REPAIR_V6.md`。

Colab GPU 配额耗尽后，在 [Kaggle 私有草稿](https://www.kaggle.com/code/yearo111/e-answer-repair-v6r1-reviewed/edit) 完成实验，结果已下载且 GPU 会话已释放。原始 30 文件和四组指标位于 `artifacts/answer_repair/v6r1_reviewed/kaggle_20261014/`；结果 ZIP 为 `answer_repair_v6r1_reviewed_outputs_export.zip`，SHA-256 `3e9cd5eec2a658cf9ff5146c2b7322c9166d35367b9af34696439847589df684`。旧 v1–v5 输出保留。128 条新预测共 123 个唯一 dataset/sample_id，各批分别报告。

从新 clone 解压冻结代码，在临时目录进行 CPU 验收并生成待人工审核队列（不需要 GPU 或重新推断）：

```sh
replay_dir="$(mktemp -d)"
tar -xzf artifacts/answer_repair/v6r1_reviewed/answer_repair_v6r1_reviewed_gpu.tar.gz -C "$replay_dir"
python3.11 scripts/audit_answer_repair_release.py --bundle-dir "$replay_dir/answer_repair_v6r1_reviewed_gpu" --results-dir artifacts/answer_repair/v6r1_reviewed/kaggle_20261014 --out-dir "$replay_dir/audit"
python3.11 scripts/prepare_answer_repair_triage.py --queue "$replay_dir/audit/pending_semantic_pairs.jsonl" --out-dir "$replay_dir/audit"
```

回放工具先验证冻结源文件与结果 manifest，再导入该归档里的验证器；不会用后续版本代码重解释历史结果。语义正确性始终单列，没有人工复评时保持 null/pending。

Git 归档清单见 [`git_archive_manifest.json`](../artifacts/answer_repair/git_archive_manifest.json)，记录每个入库实验文件的大小与 SHA-256。v5/v6r1 保留可直接阅读的原始输出、回放报告和审核记录；早期 v1–v3 保留冻结输入包及原始导出，v4/Controller 保存在 `session_20261013/E_answer_repair_20261013_results.zip`。模型权重、缓存、解压出的重复源树和下载中间件不入库。历史冻结包中的“未提交”状态与原机器绝对路径是当时的记录，保持原字节，不代表当前分支的发布状态。输入包包含本轮开发样本及 C1 证据，不是 FEVER/NQ 全量数据集。

人工阅读入口已补全代理判断：`artifacts/answer_repair/v6r1_reviewed/assisted_review_20261014/assisted_review.md`，128 条均有理由与证据定位，含 123 个唯一样本。`assisted_pairs.jsonl` 的 agent_review 保存建议，pair_review 仍全部待人工确认；原始 `release_audit/pending_semantic_pairs.jsonl` 保留不改。代理暂判去重后 9 个修复、3 个误修，7 个样本有未决字段；NQ 15 条 EM 退步暂判均为有据扩写。详细分批结果和参考错配见最新报告，不能将代理判断写成人工指标。

## 接口示例

```python
from src.compensation.controller_adapter import generation_repair
from src.compensation.qwen_repair import QwenRepairGenerator

# 上游负责构造 request 和已完成 Probe 的 decision；服务期间复用模型。
generator = QwenRepairGenerator()
result = generation_repair(request, decision, generator,
                           strategy="extractive_reanswer")  # NQ 实验
```

```python
from src.compensation.nli_repair import MNLIVerifier
verifier = MNLIVerifier()
result = generation_repair(request, decision, strategy="fever_nli",
                           verifier=verifier)  # FEVER 独立模型实验
```

两者均为显式选择的新实验选项，没有修改上游路由或静默晋升默认策略。`independent_support_status=nli_entailment/nli_contradiction` 表示独立模型的判断，仍不是人工真值。返回原答案也不意味着原答案已获证实。

## 当前可写入报告的结论

v5 同模型复核增加成本但没有提高冻结正确数；不能把六个 yes 当作语义支持证明。v4 在审核后的 FEVER 14 条上 4→6，NQ 的 2→4 来自已确认正确对照的措辞变化。v6r1 FEVER 审核后 4→7，但 dev50 45→44；NQ 审核后 EM 2→0、dev50 20→7。引文来源合法和 NLI 高分都不足以保证支持结论。保留全部负结果，两种新策略均未晋升默认。完整成本及案例见最新结果报告。

工程与本轮开发实验已经可交接；人工复评完成后再决定下一版方案，团队的共同 held-out 与三个补偿操作整合另行验收。不会继续在这批开发材料上追调阈值来宣称 E 已有效。

## v7 交接（2026-10-14 冻结，未运行）

- 协议：`ANSWER_REPAIR_V7.md`。策略 `short_answer_quote`（NQ）与 `fever_nli_anchored`（FEVER），模型、量化、阈值与 v6 相同。
- 实验包：`artifacts/answer_repair/v7_reviewed/answer_repair_v7_reviewed_gpu.tar.gz`，SHA-256 `b8f995f164783244c5c94ad591ab0f8a9a00adb1ca86a9926b83e0c2b352282f`；manifest `configs/splits/answer_repair_reviewed_v7.json`。三个批次：reviewed 28、dev50 ×2、heldout（FEVER 267、NQ 146，E 从未看过，只跑一次）。
- Notebook：`notebooks/answer_repair_colab_v7_reviewed.ipynb`、`notebooks/answer_repair_kaggle_v7_reviewed.ipynb`。本地已在解包副本中通过 99 项测试和 6 组预检。
- 设计阶段证据：`artifacts/answer_repair/v7_design/fever_anchored_replay.md`，用 v6r1 的窗口概率回放 v7 规则，FEVER 审核后 4→10、dev50 45→46、无误修。这是看着样本设计的规则在同批样本上的结果，不能当效果证据。
- 运行后：`scripts/audit_answer_repair_release.py` 回放三个批次；NQ 的有/无锚点对比可由 `raw_output` 离线重算；成对语义审核沿用 v6r1 的四项与确认清单的完整性口径。
- 第 4 批 `heldout3216`：300 条 NQ（120 utilisation / 35 accessibility / 100 correct / 45 insufficiency），gold 取自工作簿 `NQ_C1_3216_合并全量.xlsx`（本地、不入库），导出文件 `data/annotations/gold/nq_heldout_3216_v1_gold.jsonl`。实验包 `artifacts/answer_repair/v7_heldout3216/answer_repair_v7_heldout3216_gpu.tar.gz`（SHA-256 `6f277579…`），notebook `answer_repair_colab_v7_heldout3216.ipynb` / `answer_repair_kaggle_v7_heldout3216.ipynb`；约 300 次 Qwen 生成。只跑一次，结果用 `audit_answer_repair_release.py` 回放后按层报告。
