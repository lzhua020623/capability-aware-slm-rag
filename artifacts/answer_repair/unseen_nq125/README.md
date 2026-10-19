# E 一次性分层挑战：输入与推理包已冻结，已提交排队

在最终版本已确定为 r3、75 条冻结包产生之后，从原内部 calibration 自动抽取 125 条。沿用第一批的 seed 42、筛查条件、ID 哈希排序及 4:1 顺序；没有逐条预览题目或答案，也没有人工挑选。

- 100 条候选：离线 EM=0，参考 passage 出现在实际 C1 Top-5。
- 25 条对照：离线 EM=1。
- calibration 中分别有 2,388／2,848 条满足上述自动条件。候选不等于 utilisation 人工标签，对照也不等于人工确认正确。
- 重新核验全部 15 个来源分片及原划分，排除已见 ID 所在的规范化问题组；125 个固定 ID 和来源指纹记录于 `manifest.json`。这些 ID 现已登记为 E 已使用的挑战集，后续不得因结果换样本。
- 最终版本、推理指纹及 75 条包均绑定 r3。GPU 包与 Notebook 已生成并提交排队，尚未收取本批 E 推理结果，也没有最终输出的人工正确性标签。

`inputs/nq125_runtime.jsonl` 仅含问题、C1 和实际 Top-5 等运行时字段；`inputs/nq125_c1.jsonl` 保留离线参考评分数据；`offline/strata.jsonl` 仅用于分层和待审登记。控制台输出仅为数量和指纹。

此处“未见”仅指 E 组件开发的既定暴露记录。团队 Diagnosis 训练用途与这些 ID 的隔离仍需协调，Wikipedia 页面／语义近重复隔离尚未建立；不能宣称整个系统独立测试。计划只新增运行 v7 和最终 r3，C1 复用，完成后不改策略。全部改答并集交由用户确认；只审改答不产生整批完整语义准确率或完整修复召回率。

推理包：`answer_repair_final_r3_nq125_gpu.tar.gz`，SHA-256 `17dcadf90d30e8fea7a4f5d0fd259f17d9c5739484eec52d7586a5855f52855f`。包内 29 个 src 文件逐一匹配最终开发包；只改评测分层识别与报告，不改推理规则。原 111 项便携检查在 Python 3.13 通过；250 条模拟记录通过完整评分与冻结 CPU 回放，模拟记录没有作为实验结果入库。

[私有输入数据集](https://www.kaggle.com/datasets/yearo111/e-nq125-challenge-frozen-r3-20261020)；[一次性运行版本 360075807](https://www.kaggle.com/code/yearo111/e-final-nq125-r3-challenge-20261020?scriptVersionId=360075807)，提交时状态 Queued，T4 ×2。C1 复用，v7 和完整 r3 各 125 条，共 250 条新输出，上限 500 次生成。输入抽样快照见 `manifest.json`／`preparation.json`，推理打包快照见 `gpu_manifest.json`／`gpu_preparation.json`，实际状态见 `cloud_status.json`。
