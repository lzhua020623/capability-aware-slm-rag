# E 5,000请求上限与恢复封存 · 2026-10-28

最新用户指令“可以直接上5000”，沿用此前明确确认的累计US$40上限。原500复用，post-pilot最多4,500次（包括原55次）；本次复用完整554条，跳过无响应第55次但保留其US$0.13211全部预留，从第56次继续，最多再调用4,445次。后续全成功也最多4,999完整候选，不补采，不保证SFT数。

`user_acceptance.json`保存原话、覆盖关系及预算；`sample_manifest.json`保存Train内4,500个post-pilot ID，与原1,500完整保持前缀一致；`execution_receipt.json`保存输入、源码及原账本导入状态。原始失败和54个响应见[超时归档](../e_teacher_expansion_timeout_20261028/README.md)。

在线预检检查全部任务和冻结指纹，不打开离线参考。API任务只含问题、C1原答、原Top-5及固定提示/schema，没有gold或AI标签。新增代码使用旧严格预检，不改旧2,000客户端和封存哈希。后续timeout提高到600秒，增加失败阶段/耗时/HTTP状态，模型与请求体不变，串行限流及零重试保持。

```bash
python scripts/run_e_teacher_5000.py
python scripts/run_e_teacher_5000.py --execute --prompt-key
```

默认无API；实际执行仅在终端隐藏输入已有密钥，密钥不保存。新运行目录为`data/processed/e_distillation_v1/expansion_5000_results_20261028/`。同时锁定旧/新目录，导入费用不重复计算，未知费用不释放；新发生未知费用仍停止，不自动跳过。不能使用旧2,000启动器恢复。

输入压缩归档与离线参考物理分开，`verification.json`给出压缩和解压指纹。训练及学生收益尚未发生；来源/长度/语义筛选、独立Development500和Selection500协议继续有效。Test11,114不进入教师或训练。完整测试与收工核对另见本目录日志和self_check。
