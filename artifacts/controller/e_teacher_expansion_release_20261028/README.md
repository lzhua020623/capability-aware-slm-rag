# 扩量放行与费用复核

`user_batch_release.json`保留用户关于100项的聊天批量确认，没有逐项网页标注或实测语义准确率。候选采集可继续，原审核快照和两个代理争议的入训隔离不改写。

`inventory_manifest.json`是选定首批预算前生成的ID库存快照：仅含Train ID和冻结AI元数据，不解码问题/证据/gold，不是已获准执行的任务包。其“pending”是生成时状态。

`cost_reconciliation.json`按2026-10-28官方短上下文价格和原500条usage估算费用，不是Azure发票；保留原保守账本。

**当前有效范围以[2,000条封存](../e_teacher_expansion_2000_v1/README.md)为准**：500复用＋最多1,500新增，US$20累计停止上限，不执行此前5,000条／US$40选项。
