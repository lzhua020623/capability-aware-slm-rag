# FEVER 冻结 900 条 C1

`collection/` 保存原 T2 900 条任务对应的真实 Top5，按 development200 / selection200 / final_candidate500 分开，尚不含新 C1 答案或 gold。manifest 固定全部 ID、题目来源、模型/提示、检索资源、源码和逐文件 SHA；`retrieval_preparation.log` 记录实际 CPU 准备。

`controller_fever_c1_gpu_v1.tar.gz` 为冻结 GPU 输入，`bundle_receipt.json` 保存整个归档与校验索引哈希；对应 notebook 在仓库 `notebooks/controller_fever_c1_v1.ipynb`。Kaggle 只需约 2 MB 输入，不需大索引。

`cloud_status.json` 只记录实际已确认的上传/提交/运行状态，不把已打包写成已推理。新答案生成后另存输出归档与验收，不覆盖 collection。最终候选答案继续封存，原 gold 不进入模型。

2026-10-24 核查 [Kaggle Version #1](https://www.kaggle.com/code/yearo111/controller-fever-c1-900-frozen-20261024) 已 **Failed**：44.5 秒，在 CPU preflight 导入 `file_sha` 时，`src.retrieval.__init__` 意外加载 FAISS 并因缺少 `faiss` 中止。77 个输入文件校验通过；Qwen 尚未初始化，**0/900** 条回答。原始日志、复现和失败凭证在 `attempt1_failure/`。界面显示的 95 个输出文件不等于 QA 结果。

`transport_r1/` 是一次明确登记的运行修正：冻结源码只改 `src/retrieval/__init__.py` 的延迟导入，900 条题目、Top5、用途划分、模型 revision、原提示、量化与输出预算全部不变。原包、manifest、notebook、Version #1 提交记录全部保留。修正版 notebook 为 `notebooks/controller_fever_c1_transport_r1.ipynb`；新清单记录父 manifest 哈希和唯一源码差异。当前 CPU 完整回归 **733 项通过**，并从实际归档解压后禁用 FAISS/Torch/Transformers/编码器导入验证 preflight，见 `transport_r1/validation/`。云端是否开始/完成生成以 `cloud_status.json` 及后续原始输出验收为准。

**Version #2 已在 1817.1 秒后因 CUDA OOM 失败**，已生成 **645/900（200/200/245）**。完整下载 ZIP、原始日志、11 个输出文件及按输入顺序/原始生成核对的覆盖报告保存在 `attempt2_oom/`。第 646 条（最终候选集第 246 条）有 12,911 个输入 token，SDPA 预填充申请额外 17.39 GiB，超过 GPU 0 的 14.56 GiB。模型参数实际只在 `cuda:0`，两张 T4 显存未自动合并；具体 SDPA kernel 未记录，不能断言某个后端已被 profiler 证实。最终候选逐题答案未展示或评分，旧结果不构成完整 900 条基线。

`prefill_r2/` 登记新的统一计算配置：全部 900 条原生 `prefill_chunk_size=256`，仍完整读取原 Top5、使用原 prompt、模型/量化、greedy 和 64 输出 token；正式采集前增加至少 13,000 token 的虚构长上下文工程检查。相对 r1 的冻结源码仅改变 C1 runner，记录完整差异。分块可能改变数值和时延，因此整批重跑，不与旧 645 条拼接，不按逐题正确性挑选输出。新 notebook 为 `notebooks/controller_fever_c1_prefill_r2.ipynb`，原包均保留。完整 CPU 回归 **741 项通过**，实际归档解压后在禁用 FAISS/Torch/Transformers/编码器的子进程通过 preflight；GPU 长输入验证以云端产物为准。

新配置已提交 **Kaggle Version #3（361681141）**，配套私有输入为 `controller-fever-c1-900-prefill-r2-20261024`。18.0 秒通过云端 collection preflight，日志确认 256/13000 设置、77 文件输入校验及 T4×2 / Torch 2.11.0+cu128 / CUDA 12.8。提交凭证为 `prefill_r2/submission_v3.json`；仅有 Running/preflight 不能视为长输入检查或 900 条采集完成。

后续日志于 **272.8 秒打印 25/900 条正式生成**，见 `prefill_r2/runtime_observation_v3.json`。冻结 runner 仅在三条虚构工程检查全部通过后开始正式采集，因此由控制流确认至少 13,000-token 虚构输入已通过；原始工程 trace 仍待最终导出核验。25 是日志下界，不能视为精确实时计数；900 条尚未完成验收。

提交状态与冻结包绑定检查新增一项回归，当前完整 CPU 回归 **727 项通过**，见 `submission_validation/`。73 个冻结源码文件与 `8d0d033` 提交一致，归档 77 个内容文件全部通过哈希核验；原 GPU 包及采集算法未改动。

执行命令、异常规则和当前范围见 [交接说明](../../../docs/CONTROLLER_FEVER_C1_V1.md)。
