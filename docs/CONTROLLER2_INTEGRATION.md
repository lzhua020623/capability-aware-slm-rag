# Controller 2 集成交接 · 2026-10-27

**当前生产 Adaptive RAG 状态与入口**见 [CONTROLLER_1024_BASELINE.md](CONTROLLER_1024_BASELINE.md)（E2E production path passed）。下文保留分支合并与兼容处理的历史交接记录。

后续操作在 `controller_2`。本轮从 `controller@5da8fa690bb04bd65f19ee9629b899d569fa0abe` 创建分支，合入两条增强路线的最新内容；下列合并说明描述当时的兼容集成，不以其中“E unavailable / P60 未交付”等过时表述为准。

| 来源 | 冻结提交 | 本轮加入 |
|---|---|---|
| `feature/retrieval-enhancement` | `37e469e0bd101dc5a905f95aeca1cafdc6a07e9d` | 3 个提交：生成回放、Windows 兼容、历史 GPU 输出、评分和路由加权审计 |
| `feature/reranking-enhancement` | `55832850640f5fe37c52a02779cc389a2e5f0276` | 18 个提交：重排训练、答案级与误修评测、随机重排/句子裁剪对照、候选留痕、统一适配器、答案选择器训练工具 |

补检索合并提交为 `5c793c7`。第二个合并提交同时包含下述兼容处理、测试与本交接；最终提交号以 Git 历史为准。

## 兼容处理

- 保留原 P3-20 的 FEVER 支持、标签解析、失败回退、调用计量与包的延迟导入。上游最新三个提交主要补充实验工具与证据，不覆盖 Controller 已有的跨数据集适配。
- `RefinementConfig/Result/Status` 以 `refinement_schema.py` 为唯一类型定义，`schema.py` 和包级旧导入继续指向同一类型。已接受结果的旧构造调用可从 final_answer 确定 candidate_answer；拒绝且未记录候选时仍为 unknown，不补造拒绝内容。
- 新轨迹保留被拒绝的候选答案；证据 ID 不变但正文被裁剪时也能触发重新生成。新能力由显式实验配置启用。
- 保留 `EnhancementResult` 对非有限延迟的拒绝，以及现有执行器对动作、答案变化和失败回退的一致性检查。
- 补检索独立运行包增加已拆分的 `src/generation/prompts.py`。Windows 原包审计仍核对批准的源码哈希；较新 checkout 不会被冒充为历史原包。
- 历史 T5 包装回归从原始冻结包恢复源码，补齐缺失文件前也核对旧哈希；新增检查确认当前 Controller 源码不能重建旧封存。生产封存检查与历史包均未放宽或覆盖。
- 保留仓库 `AGENTS.md`、旧导入回归与全部历史实验归档，合并 `.gitignore` 两侧规则；用户未提交的辅助审核文件不纳入本轮提交。

## 当前实际运行配置

`configs/controller/current_baseline.json` 和 `p60_online_adapter.json` 保持合并前字节。

- 后续基线仍为 10/24 Diagnosis/P60/P64，Probe OFF。
- 原始 78 维在线特征提供器、三个 HGB 的训练/调参/校准/分析 ID 仍未交付，生产预检查退出 2；不调用虚构特征或退回旧路由。
- NQ/FEVER 的 E 仍为 unavailable。蒸馏、LoRA、Azure 教师调用和新的 GPU 运行均未启动。
- 默认重排仍使用已登记的 `BAAI/bge-reranker-large`，保留 Top-5，不启用裁剪。新训练重排权重未交付/登记；答案选择器目前有训练脚本和数据统计，没有已验收模型。
- 不将重排的排序指标、随机对照或补检索的路由加权估计当作当前集成系统收益。来源结果在各自报告中解释，跨环境绝对成绩不直接拼接。

## 本地验证与可选依赖

专项检查与完整回归记录见 [验证归档](../artifacts/controller/controller2_integration_v1/README.md)。这些测试使用虚构问题/提供器、历史冻结资产与本地审计，不进行教师 API 调用或 GPU 推理。

新的随机句子裁剪对照使用 NLTK 的英文句子切分器，需在现有项目依赖之外显式准备：

```bash
python -m pip install -r requirements-refinement-experiments.txt
python -m nltk.downloader punkt_tab
```

模型代码不在运行时自动下载此资源；需要离线复现时，提前准备资源并配置 `NLTK_DATA`。本轮测试环境和资源校验记录在归档中。原 `requirements.txt` 已声明 pandas/openpyxl，完整测试环境也补齐这些依赖。

## 后续任务

1. 接入团队 `feature/baseline-evaluation@387ddaf` 的原始 NQ 划分，固定 Train 51,866、Validation 11,117、Test 11,114；取得 matched RAG 的 predictions/metrics/run_config，按段落 ID 回读证据。该第三分支不在本次两路线合并范围内。
2. 登记历史曝光与跨分支训练 ID。旧 E 登记 69 条、Controller 开发 34 条、选型 27 条落入团队 test，三者合计至少 130 条；不能将全 test 称为 E 完全未见。旧保留池继续保护，FEVER 另行统一。
3. 核对新重排器和答案选择器的数据生成/训练 ID 与团队划分；现有脚本的局部划分不能替代团队 70/15/15。权重、近重复隔离、数据来源与预算通过核查后再登记候选。
4. 并行补齐 P60 两项材料，并准备 E 的短输出契约与教师资格验证。E 训练只使用允许的 Train 数据，Validation 用于选型；Test 不进入训练或提示设计。
5. 在固定模块/路由配置下验证 E 候选，完成工程和真实效果验收，再冻结新版本与最终团队配对评测协议。

旧 T1–T5 包、成绩和停止结论保留原样。旧各 500 条最终候选属于原协议，团队新 test 的范围、比较组、GPU 预算与统计方法需另行封存。主线进度见 [T1–T9](CONTROLLER_TASKS.md)。
