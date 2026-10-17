# E 的 Colab 与 Kaggle 实验

当前工作基础是已完成的 v7，下一轮 v8 尚未实现。下面保留 v7 的复现入口；历史 v1 至 v6r1 的运行过程与旧环境状态见 [历史执行记录](history/answer_repair/GPU_RUNS_V1_V6R1.md)。已有结果无需重新推理，优先离线回放。

## v7 已完成实验

| 实验 | Notebook | 输入与结果 |
|---|---|---|
| 主包：审核集、dev50、预实验池留出集，共 541 条 | [Colab](../notebooks/answer_repair_colab_v7_reviewed.ipynb) / [Kaggle](../notebooks/answer_repair_kaggle_v7_reviewed.ipynb) | `artifacts/answer_repair/v7_reviewed/` |
| 第 4 批：NQ3216 分层留出集，300 条 | [Colab](../notebooks/answer_repair_colab_v7_heldout3216.ipynb) / [Kaggle](../notebooks/answer_repair_kaggle_v7_heldout3216.ipynb) | `artifacts/answer_repair/v7_heldout3216/` |

两批均已在 Kaggle 运行一次。实验设计见 [冻结协议](ANSWER_REPAIR_V7.md)，完整结果、局限和后续路线见 [v7 报告](ANSWER_REPAIR_RESULTS_20261014_V7.md)。留出集不作为下一轮调参开发集，复用 Notebook 不能记作新的算法版本。

每个版本目录保留输入 `.tar.gz`、原始输出 ZIP、`kaggle_20261014/` 和 `release_audit/`。准确哈希以 [归档清单](../artifacts/answer_repair/git_archive_manifest.json) 为准；解压副本 `gpu_bundle/` 可在需要时重建，不入库。

## 无 GPU 离线回放

从仓库根目录执行，把输入包解压到新的临时目录，再用包内冻结代码核对保存结果：

```sh
replay_dir="$(mktemp -d)"
tar -xzf artifacts/answer_repair/v7_reviewed/answer_repair_v7_reviewed_gpu.tar.gz -C "$replay_dir"
python3.11 scripts/audit_answer_repair_release.py \
  --bundle-dir "$replay_dir/answer_repair_v7_reviewed_gpu" \
  --results-dir artifacts/answer_repair/v7_reviewed/kaggle_20261014 \
  --out-dir "$replay_dir/audit"
```

第 4 批使用对应的 `v7_heldout3216` 包与结果目录，输出到另一个临时目录。回放检查冻结源文件、结果 manifest、裁决和分数，不加载模型，不自动填写语义人工标签。

当前人工审核统一使用 [v7 总表](../artifacts/answer_repair/v7_combined_review/review_all.txt)，结构化数据在同目录 `review_all.jsonl`；新生成的空审核队列不覆盖已有人工作表。

## 后续新实验

先完成新策略开发、固定开发样本、对照和调用预算，再生成独立实验包及 Notebook。使用新输出目录，不覆盖旧包。历史构建脚本的默认输出已指向历史 Notebook 目录；新实验必须显式指定版本和输出位置。此轮仓库整理没有更改模型、提示词、阈值、策略或样本。
