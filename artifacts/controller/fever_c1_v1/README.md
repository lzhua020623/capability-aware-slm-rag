# FEVER 冻结 900 条 C1

`collection/` 保存原 T2 900 条任务对应的真实 Top5，按 development200 / selection200 / final_candidate500 分开，尚不含新 C1 答案或 gold。manifest 固定全部 ID、题目来源、模型/提示、检索资源、源码和逐文件 SHA；`retrieval_preparation.log` 记录实际 CPU 准备。

`controller_fever_c1_gpu_v1.tar.gz` 为冻结 GPU 输入，`bundle_receipt.json` 保存整个归档与校验索引哈希；对应 notebook 在仓库 `notebooks/controller_fever_c1_v1.ipynb`。Kaggle 只需约 2 MB 输入，不需大索引。

`cloud_status.json` 只记录实际已确认的上传/提交/运行状态，不把已打包写成已推理。新答案生成后另存输出归档与验收，不覆盖 collection。最终候选答案继续封存，原 gold 不进入模型。

2026-10-24 核查 [Kaggle Version #1](https://www.kaggle.com/code/yearo111/controller-fever-c1-900-frozen-20261024) 已 **Failed**：44.5 秒，在 CPU preflight 导入 `file_sha` 时，`src.retrieval.__init__` 意外加载 FAISS 并因缺少 `faiss` 中止。77 个输入文件校验通过；Qwen 尚未初始化，**0/900** 条回答。原始日志、复现和失败凭证在 `attempt1_failure/`。界面显示的 95 个输出文件不等于 QA 结果。

`transport_r1/` 是一次明确登记的运行修正：冻结源码只改 `src/retrieval/__init__.py` 的延迟导入，900 条题目、Top5、用途划分、模型 revision、原提示、量化与输出预算全部不变。原包、manifest、notebook、Version #1 提交记录全部保留。修正版 notebook 为 `notebooks/controller_fever_c1_transport_r1.ipynb`；新清单记录父 manifest 哈希和唯一源码差异。当前 CPU 完整回归 **733 项通过**，并从实际归档解压后禁用 FAISS/Torch/Transformers/编码器导入验证 preflight，见 `transport_r1/validation/`。云端是否开始/完成生成以 `cloud_status.json` 及后续原始输出验收为准。

提交状态与冻结包绑定检查新增一项回归，当前完整 CPU 回归 **727 项通过**，见 `submission_validation/`。73 个冻结源码文件与 `8d0d033` 提交一致，归档 77 个内容文件全部通过哈希核验；原 GPU 包及采集算法未改动。

执行命令、异常规则和当前范围见 [交接说明](../../../docs/CONTROLLER_FEVER_C1_V1.md)。
