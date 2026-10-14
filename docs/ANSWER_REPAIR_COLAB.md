# E 的 Colab / Kaggle 实验

2026-10-14（v7，主包已运行一次，结果 ZIP SHA-256 `48e82e600f189b2b5e7f98063883ffa3b14574c26debfc50d011170f7308d5d1`，见 [v7 结果](ANSWER_REPAIR_RESULTS_20261014_V7.md)；第 4 批未运行）：[Colab v7](../notebooks/answer_repair_colab_v7_reviewed.ipynb) / [Kaggle v7](../notebooks/answer_repair_kaggle_v7_reviewed.ipynb)，配套 `artifacts/answer_repair/v7_reviewed/answer_repair_v7_reviewed_gpu.tar.gz`（SHA-256 `b8f995f164783244c5c94ad591ab0f8a9a00adb1ca86a9926b83e0c2b352282f`，校验清单索引 `f421fd5378b9ac419714de51905aaa65769457b4231195eb53cc6995de67dc12`）。协议见 [v7](ANSWER_REPAIR_V7.md)。7 个代码单元格：校验、依赖与 99 项测试、28 条审核后样本预检、审核后样本运行、dev50 与 held-out 运行、汇总、导出。held-out 只运行一次。预计 Qwen 生成 14+50+146=210 次（T4 约 25 分钟），NLI 331 条。第 4 批 [Colab](../notebooks/answer_repair_colab_v7_heldout3216.ipynb) / [Kaggle](../notebooks/answer_repair_kaggle_v7_heldout3216.ipynb)，配套 `artifacts/answer_repair/v7_heldout3216/answer_repair_v7_heldout3216_gpu.tar.gz`（SHA-256 `6f27757935128f3e04e2881e1fb832885b10d45ef55316fa425b889950651bde`），300 次 Qwen 生成，只跑一次。运行后用 `scripts/audit_answer_repair_release.py --bundle-dir <解压目录> --results-dir <结果目录> --out-dir <审计目录>` 回放三个批次。

2026-10-14：v5 的 112 条和 v6r1 的 128 条真实预测均已归档并通过冻结代码回放，见 [结果报告](ANSWER_REPAIR_RESULTS_20261014.md)。最新可复现版本是 [Colab v6r1](../notebooks/answer_repair_colab_v6r1_reviewed.ipynb) 或 [Kaggle v6r1](../notebooks/answer_repair_kaggle_v6r1_reviewed.ipynb)，配套 `artifacts/answer_repair/v6r1_reviewed/answer_repair_v6r1_reviewed_gpu.tar.gz`。运行协议见 [v6](ANSWER_REPAIR_V6.md)。Colab GPU 配额耗尽后，已在 [Kaggle 私有草稿](https://www.kaggle.com/code/yearo111/e-answer-repair-v6r1-reviewed/edit) 完成所有单元格，48 文件校验、81 项云端测试及四组推断通过。输出 30 文件已下载，GPU 会话已释放。

最新结果目录：`artifacts/answer_repair/v6r1_reviewed/kaggle_20261014/`；ZIP SHA-256 `3e9cd5eec2a658cf9ff5146c2b7322c9166d35367b9af34696439847589df684`。人工队列为 `v6r1_reviewed/release_audit/review_priority.md`（阅读）与 `review_priority.jsonl`（可编辑），均未完成人工判断。无需为已有结果重新推断。

v6r1 归档 SHA-256：`74d40b01c96245e1caf2ac8af786bf3b05394971863bddc81a75a2a2afa45ef5`。Notebook 含 7 个代码单元格，依次完成校验、依赖与 81 项测试、28 条审核后开发样本、原 dev50 × 2 回归和结果导出。新旧包有各自独立输出目录。FEVER/NQ 策略各自只作用于对应数据集，分类器成本单独报告。不要重跑现有 smoke notebook 的所有历史单元格。

Kaggle 会自动解压上传的 `.tar.gz`。专用 notebook 同时支持原归档和已解压目录：后者先核对 `bundle_checksums.json` 的固定 SHA-256 `1c095e9fb52077898e8eb85672f76ba4ac052e87b9e099da61df523be404d0bd`，再验证全部 48 个文件，并保存 `input_integrity.json`。预解压模式不声称重新验证原归档字节。此适配仅改变文件加载，不改冻结实验代码；本地已验证两种加载路径及文件/校验清单篡改拒绝。验证记录位于 `artifacts/answer_repair/v6r1_reviewed/kaggle_transport_validation.json`。

导出结果后，本地只用冻结包里的代码进行回放：

```sh
python3.11 scripts/audit_answer_repair_release.py --bundle-dir artifacts/answer_repair/v6r1_reviewed/gpu_bundle --results-dir artifacts/answer_repair/v6r1_reviewed/kaggle_20261014 --out-dir artifacts/answer_repair/v6r1_reviewed/release_audit
```

以下为 v4 历史执行记录。

v4 便携包为 `artifacts/answer_repair/v4/answer_repair_v4_gpu.tar.gz`，对应 [v4 notebook](../notebooks/answer_repair_colab_v4.ipynb)。真实实验在 [Colab T4](https://colab.research.google.com/drive/1XI0qoiW0NIQHiD7iOX8ewqM07Y-snxMQ) 执行；本地代码尚未提交或推送。

v4 包 SHA-256：`2a40d4fbaff591b02d116243f92404731cb71ccc5ce8a77092f4fda70dd3156e`。历史 v1/v2/v3 的包和输出分别保留；不同代码版本使用独立输出文件。

1. Colab 用“文件 → 上传笔记本”打开 v4 notebook，选择 T4 GPU。Kaggle 可导入 notebook，启用 GPU 和 Internet，将 v4 包添加为私有输入。
2. 执行 CUDA 检查，上传 v4 包。Notebook 会验证归档哈希及各个代码/输入文件的哈希。
3. 执行依赖安装、环境检查、54 项 CPU 测试及固定样本预检。首次真实调用会下载 Qwen2.5-7B-Instruct，后续复用云端 cache。
4. 在 FEVER/NQ 各 10 条上比较 `verify_repair` 与 `grounded_reanswer`，共 40 次额外生成，审核所有输出。
5. 可选开发格默认 `RUN_DEVELOPMENT=False`。审核后可运行选定 E 策略与 `reanswer` 对照，各数据集 50 条，共 200 次生成。候选策略在 `SELECTED_E_STRATEGY` 中显式选择。
6. 最后一格导出预测、汇总、环境、输入 manifest、文件校验和日志 ZIP。审核模板中的人工栏为空，生成模板不表示审核已经完成。

只做逐条 NF4 推理，无需重建检索库。Top-5 内容/顺序固定，每条 E 至多一次调用、384-token 上限。参考答案仅参与离线评分。NQ 引用字面匹配和 answer span 检查都不等于语义支持验证。

输出目录是 Colab 的 `/content/answer_repair_v4_outputs` 或 Kaggle 的 `/kaggle/working/answer_repair_v4_outputs`。不自动挂载 Drive。同一运行时内可续跑；运行时被释放后，需要重新上传包并恢复旧输出。释放前先导出结果。

## 已完成的工程 smoke

2026-10-13 的真实 T4 v4 各 10 条结果：

| 数据集 | C1 正确 | Verify 正确 / invalid | 独立候选正确 / invalid |
|---|---:|---:|---:|
| FEVER | 7/10 | 7/10，2 条 invalid | 8/10，0 条 invalid |
| NQ EM | 2/10 | 2/10，4 条 invalid | 2/10，4 条 invalid |

FEVER 的那一条标签改对引用了委内瑞拉 Valencia Municipality，不能证明目标实体及 administrative centre 的人口命题；不作为可靠的 grounded repair 成功。NQ 独立候选改了两条文本，但 EM 无增益，胫骨/腓骨样本仍回答了单个骨骼的相对位置。引用检查合法的 keep 也存在引用无关实体的情况。

v4 已修复完整 JSON 末 token 附带反引号的问题：保存原始输出及实际提交解析的对象文本，严格 schema 和 quote 检查保持不变。固定 50 条开发结果已经完成：FEVER 45→46，但新增正确是实体不匹配的 Valencia；NQ 20→20，改对 1 条、改错 1 条。Reanswer 两套均无增益。smoke 是该开发集的子集，不作为独立第二批数据合并统计。完整输出、成本、案例和包哈希见 [结果报告](ANSWER_REPAIR_RESULTS_20261013.md)。

## 重新生成 notebook

```bash
python3.11 scripts/build_answer_repair_notebook.py \
  --bundle artifacts/answer_repair/v4/answer_repair_v4_gpu.tar.gz \
  --out notebooks/answer_repair_colab_v4.ipynb \
  --strategy grounded_reanswer
```

生成器嵌入归档哈希，并对 8 个代码单元格做语法检查。已存在的归档拒绝被不同代码覆盖，下一版算法使用新的包名及目录；旧包内文档也是当时的代码快照。

## Controller 与 E 整合

E 对最终 `ControllerDecision` 的执行接口见 [ANSWER_REPAIR.md](ANSWER_REPAIR.md)。上游分支冻结于 `7c7863f98f87e892119481a79e23ca6dcebdced0`。隔离整合包 `artifacts/answer_repair/e_controller_gpu.tar.gz` 复用该分支的 Diagnosis、Probe、Safe Binary、Controller 与 v4 E；不重训或调阈值。

它只验证 Controller→E：另外两类增强动作仅记录，不执行。Probe 与 E 共用一份 NF4 模型，分别记录调用/token/生成延迟。固定开发样本来自训练交集，不能将该结果当作 held-out 或完整 adaptive system 效果。

实际整合已在 T4 跑完固定 100 条：27 次 Probe、1 次 E；FEVER / NQ 正确数均未改变。代码、原始输出与运行日志已导出并逐条重放校验。组合下载包为 [E_answer_repair_20261013_results.zip](../artifacts/answer_repair/session_20261013/E_answer_repair_20261013_results.zip)。
