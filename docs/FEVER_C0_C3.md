# FEVER500：7B C0 / C3 运行说明

本分支执行冻结的 Preliminary Recoverability Experiment 中的 FEVER C0 和 C3。
研究目标是判断 Small-Model RAG 的错误能否通过正确证据修复；后续需汇合相同 ID 的 C1 结果，才能计算 Recoverable Failure Rate。仅 C0/C3 的 accuracy 不能代替该统计。

## 固定协议

- 配置：`configs/preliminary.yaml`，模型参数同时检查 `configs/base.yaml`。
- 样本：`configs/splits/preliminary_fever_500.json`，500 个唯一 ID，250 SUPPORTS + 250 REFUTES，保持 manifest 顺序，不重新抽样。
- 模型：`Qwen/Qwen2.5-7B-Instruct`，4-bit NF4，greedy，`max_new_tokens=64`，seed 42。
- C0：仅 instruction + claim，模型输入不含证据或 gold label，结果 `evidence=[]`。
- C3：instruction + 当前样本全部 `gold_evidence_text` + claim，沿用共享 FEVER evidence 格式，不截成 Top-5、不检索。
- 评测：沿用 `src/evaluation/fever.py` 的标签解析，计算 classification accuracy；UNKNOWN 计错。Smoke 若输出无法解析的标签则报错。

原有 NQ/FEVER manifests、共享配置、提示词文本和解码参数均未修改。为支持无模型依赖预检，提示词构造移至 `src/generation/prompts.py`，旧的 `src.generation.qwen` 导入入口继续可用。

## 租卡前准备

先取得组内已预处理的 `data/processed/retrieval/fever_dev.jsonl`，放到相同路径。
该文件必须包含冻结 ID 对应的 claim、label、gold_page_ids 和 gold_evidence_text。
它被 gitignore，**clone 不会下载它**。C0/C3 不需要 FEVER corpus、FAISS 索引、BGE 或 NQ 数据。
如需从原始数据重建，参考 [SETUP.md](SETUP.md)；现有完整预处理脚本也会处理 NQ 和 FEVER corpus。

在仓库根目录执行；预检环境只需 Python 和 PyYAML：

```bash
python -m pip install pyyaml
python scripts/run_fever_preliminary.py --check-only
```

预检验证模型配置、manifest、源数据，以及已有结果是否允许续跑。缺数据会明确失败，不会自动抽样或下载模型。

## GPU 机器执行

```bash
git clone --branch feature/F500-c0c3 https://github.com/lzhua020623/capability-aware-slm-rag.git
cd capability-aware-slm-rag
```

将上述 `fever_dev.jsonl` 放入指定路径，创建 Python 环境，按 [SETUP.md](SETUP.md) 安装适合机器的 CUDA PyTorch 和 `requirements.txt`。已有仓库则切换到该分支并 `git pull --ff-only`。

依次执行，前一步通过再继续：

```bash
python scripts/run_fever_preliminary.py --check-only
python scripts/run_fever_preliminary.py --smoke-test
python -u scripts/run_fever_preliminary.py
```

Smoke 对第一个冻结样本执行 C0 和 C3，不写正式结果；模型预测错不代表工程失败。
正式运行共 1000 次生成，逐条落盘；中断后执行同一命令即可跳过已完成记录。
也可以分别执行 `--conditions C0` 或 `--conditions C3`。同一结果文件只运行一个写入进程。

输出：

- `results/preliminary/fever_c0_7b.jsonl`：500 条。
- `results/preliminary/fever_c3_7b.jsonl`：500 条。

结束时输出总 accuracy、两类各自 accuracy、UNKNOWN 数量。每条结果包含原始输出、证据、耗时和实验指纹，便于与 C1 按 sample_id 汇合。
再次运行已完成的条件，只检查并汇总结果，不加载模型。

## 续跑与结果保护

实验指纹绑定冻结样本内容、提示词、模型配置及推理/评测代码。旧模型、不同数据/提示词、无指纹的旧结果、重复 ID 或指标不一致都会拒绝续跑。需要新跑时先将旧结果移到独立归档目录，不要混写。

结果每条写入后 flush + fsync；末条 JSON 完整但缺少换行时，会补换行后追加。如断电留下不完整 JSON，脚本会报告具体文件和行号并停止，保留原文件。先备份，确认仅末行是未完成写入后移除该末行，再续跑；不要忽略中间损坏行。

续跑使用同一软件环境；指纹不锁定 Python 包版本或远端模型 revision。建议运行时保存 `python -m pip freeze` 和当前 `git rev-parse HEAD`，与结果一同交接。

## 本地验证范围

```bash
python -m unittest discover -s tests -v
```

回归测试使用真实冻结 ID 与合成 claim/evidence、模拟生成器，在临时目录验证 500 × 2 的中断续跑、输入隔离、数据校验和结果防混写；不会生成正式实验结果。
本分支在无 CUDA、无正式 `fever_dev.jsonl` 的环境完成代码验证。正式数据预检与真实 7B NF4 smoke 仍须在数据和 GPU 就绪后执行。
