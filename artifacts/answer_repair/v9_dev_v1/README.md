# v9 首版工程快照（失败，完整保留）

- 输入：`answer_repair_v9_dev_v1_gpu.tar.gz`，SHA-256 `4f3e3e2acf4de62bf70089b2a5ee13b66afc917641c8f602f6dc6faaa79046bf`。
- Kaggle Version 1：`360999513`，Private，T4×2，592.9 秒后 CUDA OOM。
- 输出 ZIP：`kaggle_20261022_v1/outputs.zip`，SHA-256 `389dce00a3efb40effe4ee28106bf17bf32a7f78a52bc219dd9c0ef0ef7bd91f`。
- `archive_manifest.json` 记录全部 92 个 ZIP 成员。完整源码与输入副本保存在原始 ZIP 和冻结输入包；工作解压目录不重复逐份入库。
- 原始 `results/`、`environment.json`、`bootstrap.json` 与运行日志保留；`local_replay.json` 记录五组完整回放、一组前缀回放；`engineering_summary.json` 记录 26／30 条保存结果及无效输出。

首版工程门槛未通过，不宣称收益。格式压缩另立 `v9_dev_v2`，不改本包字节。完整说明见 [工程报告](../../../docs/ANSWER_REPAIR_V9_ENGINEERING_20261022.md)。
