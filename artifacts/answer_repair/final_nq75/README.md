# E 最终 75 条开发实验

最终推理版本为 r3。最后 r4 的三项预声明回归仅 N12 通过，按原规则回退 r3；没有根据 EM 选择版本，也不追加 r5。

原 75 条为 11 个已确认内部 E 目标和 64 个正确对照。C1 复用，新增 v7／完整 r3／去原检查／去候选检查四组各 75 条；共 300 条输出，上限 600 次生成。仅用于最终开发描述，完成后不再调参，不当作未见收益证据。

- 输入归档：`answer_repair_final_r3_nq75_gpu.tar.gz`。
- SHA-256：`b5be903db6efecb5681498f0110ebed65920369ca1e5d466492ebae7455fc98b`。
- 推理源码：29 个文件与选定 r3 冻结源逐一一致；只有评测清单识别和打包脚本等非推理文件按记录适配。
- 测试：Python 3.13 通过该冻结版本原 111 项便携检查。AST 兼容适配器只在 unittest 进程运行，不修改冻结包或模型进程。
- Notebook：`notebooks/answer_repair_final_nq75.ipynb`。
- 私有输入：<https://www.kaggle.com/datasets/yearo111/e-final-nq75-frozen-r3-20261020>。
- 运行版本：<https://www.kaggle.com/code/yearo111/e-final-nq75-r3-comparison-20261020?scriptVersionId=360070600>，T4 ×2。

`manifest.json`、`preparation.json` 是冻结准备快照，`validation.json` 是本地校验；实际云端状态见 `cloud_status.json`。目前只确认已提交并显示 Running，尚无 75 条结果或人工最终收益。
