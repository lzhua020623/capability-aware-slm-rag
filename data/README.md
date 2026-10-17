# Data

项目使用 Natural Questions 和 FEVER；FinDER 仅为可选方向，当前没有对应正式流水线。

| 目录 | 内容 | Git 管理 |
|---|---|---|
| `raw/` | 官方下载、FEVER Wikipedia 和 NQ 原始数据 | 忽略数据内容，保留 `.gitkeep` |
| `processed/` | controlled 样本、检索 corpus 和索引等可重建产物 | 忽略数据内容，保留 `.gitkeep` |
| `annotations/assisted/` | 代理辅助审核判断和理由 | 已入库，不能当人工真值 |
| `annotations/human/` | 用户确认、成对人工字段及其来源 | 已入库，部分记录仍待审 |
| `annotations/gold/` | 第 4 批 300 条 NQ 的冻结参考答案 | 已入库，仅离线评测使用 |
| `annotations/nq_train_delivery/` | 新 NQ Train 分片清单、来源说明及 Drive 核对报告 | 小型元数据入库；原始 15 分片在本地 `processed/external/` |

2026-10-17 已接入 NQ Train 全部 74,097 条 C1，标签全部为 null。全量 E 兼容输入在本地 `processed/answer_repair/nq_train_v1/c1_scored.jsonl`，受 Git 忽略；首批 100 条开发审核材料与内部划分配置入库，见 [接入说明](../docs/ANSWER_REPAIR_NQ_TRAIN.md)。100 是审核批次，不是数据集总量，筛查分组不等于人工 diagnosis 标签。

本地 `.xlsx` 工作簿被忽略，但可能包含尚未完整导出的上游标注和 gold 来源，应保留可信备份。原始数据未上传 GitHub，不等于仓库没有标注或参考答案。

NQ、FEVER 各 500 条项目样本和 NQ3216 扩展池的 normalized diagnosis 标签位于 Diagnosis 分支，详见 [项目进度](../docs/PROJECT_STATUS_AND_DIRECTION.md)。上游 AI 初标与人工确认必须区分。数据获取与处理见 [安装说明](../docs/SETUP.md)；E 运行时只使用 query、C1 初始答案和检索证据，不读取 gold。
