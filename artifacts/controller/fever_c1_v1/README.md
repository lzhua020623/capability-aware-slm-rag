# FEVER 冻结 900 条 C1

`collection/` 保存原 T2 900 条任务对应的真实 Top5，按 development200 / selection200 / final_candidate500 分开，尚不含新 C1 答案或 gold。manifest 固定全部 ID、题目来源、模型/提示、检索资源、源码和逐文件 SHA；`retrieval_preparation.log` 记录实际 CPU 准备。

`controller_fever_c1_gpu_v1.tar.gz` 为冻结 GPU 输入，`bundle_receipt.json` 保存整个归档与校验索引哈希；对应 notebook 在仓库 `notebooks/controller_fever_c1_v1.ipynb`。Kaggle 只需约 2 MB 输入，不需大索引。

`cloud_status.json` 只记录实际已确认的上传/提交/运行状态，不把已打包写成已推理。新答案生成后另存输出归档与验收，不覆盖 collection。最终候选答案继续封存，原 gold 不进入模型。

执行命令、异常规则和当前范围见 [交接说明](../../../docs/CONTROLLER_FEVER_C1_V1.md)。
