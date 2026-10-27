# E 2,000候选采集封存 · 2026-10-28

最新用户指令：**先2000条 有效再扩量**。本轮是累计2,000候选（原500复用＋最多1,500新增），累计单价估算停止上限US$20；此前5,000／US$40选择已被覆盖。按原500条usage外推累计约US$13.71，其中新增约US$10.28；未核实Azure实际账单。新增付费调用0，训练0，学生效果尚未测量。

- `budget_acceptance.json`：最新原话、覆盖关系、预算和不自动扩量的边界。
- `sample_manifest.json`：原500 ID＋本轮1,500 ID，四类AI元数据各375，属于Train；不根据gold/EM抽样。
- `execution_receipt.json`：输入文件、原始15分片、源码和协议指纹。执行前全部匹配；原C1/Top-5不变。
- `teacher_tasks.jsonl.gz`：本轮1,500个运行时任务的无损压缩归档。只有问题、原答、Top-5及固定提示/输出schema，不含离线参考字段。
- `offline_references.jsonl.gz`：物理隔离的离线Train参考，不进入教师请求；不是监督目标。
- `verification.json`：逐条与原C1分片独立匹配、用途/交叉检查、压缩往返、限流包大小及归档指纹。没有调用Azure。
- `full_tests.log`、`self_check.json`：完整回归和收工检查（测试完成后写入）。

恢复输入时只把`teacher_tasks.jsonl.gz`解压到配置的`inputs_directory/teacher_tasks.jsonl`；在线预检不打开参考文件。离线审核另将`offline_references.jsonl.gz`解压到同目录的参考文件。先比对`verification.json`中的压缩/解压SHA-256，再运行：

```bash
python scripts/run_e_teacher_expansion.py
```

实际采集需显式`--execute`，密钥由用户在本机终端隐藏输入。客户端使用独立进程锁和追加式journal；原500不重发。发送前完整预留，按固定单价估算结算；未知费用或版本变化停止且不自动重试。已付费但目标契约不合格则隔离、继续下个登记ID，不补采。最多新增1,500次，不能自动转为5,000条。

2,000候选不等于2,000条合格SFT。原100项按聊天批量放行候选采集，不生成不存在的网页逐条判断或100%语义准确率；原试产审核快照不改写。两项代理争议保留临时入训隔离，目标还需来源、384-token长度及语义筛选。只训练E的Qwen adapter，C1不改。是否继续扩量以训练后学生Development500的配对净收益和误修审核为依据；Selection500只给最终候选一次确认，团队Test11,114条保持封存。

相关实现和用法见[扩量与费用说明](../../../docs/E_TEACHER_EXPANSION_20261028.md)。历史5,000个ID库存仅作离线准备快照，不构成本轮API授权；原500结果和保守账本不覆盖。
