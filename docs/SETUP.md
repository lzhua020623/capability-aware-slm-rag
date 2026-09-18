# Setup

Windows 环境。PyTorch **不在** `requirements.txt` 里，需按本机硬件单独安装，再装其余依赖。

模型和原始/预处理数据都很大，**不会提交到 GitHub**（见 `.gitignore`：`.venv/`、`data/raw/*`、`data/processed/*`、`models/`、`checkpoints/`）。clone 之后用下面已验证的脚本在本地重建数据。Preliminary Recoverability Experiment 必须使用冻结的 sample manifests，不要重新抽样。`results/` 里体积较小的 smoke-test / retrieval metrics 可以随仓库保留，用于对照，不是可执行的数据或模型。

当前正式 Retriever：**BGE + FAISS + Top-5**（`BAAI/bge-base-en-v1.5`）。Primary SLM：`Qwen/Qwen2.5-7B-Instruct`，4-bit NF4。第一次加载该模型时，Hugging Face 会把它下载到本地 cache（不在本仓库里）。

**smoke test** = 工程验证（工程能否跑通）。  
**Preliminary Recoverability Experiment** = 配置已冻结，实验尚未开始。见 [PRELIMINARY_EXPERIMENT.md](PRELIMINARY_EXPERIMENT.md)。

## 1. Clone repository / 克隆存储库

```powershell
git clone <repository-url>
cd capability-aware-slm-rag
```

## Shared Experiment Data Setup

组员 clone 仓库、创建 Python 环境并安装依赖后，用仓库里已验证的脚本在本地重建共用数据，不需要 `ELEC5623_shared_data.zip`：

```powershell
python scripts/download_data.py
python scripts/verify_data.py
python scripts/prepare_nq.py
python scripts/prepare_retrieval_data.py
```

负责 NQ C1 的组员可用已记录的 NQ 检索脚本生成正式 BGE IndexFlatIP：`python scripts/run_nq_bge_retriever.py`。

所有人必须使用冻结的 manifests，不得重新抽样：

- NQ：`configs/splits/preliminary_nq_500.json`
- FEVER：`configs/splits/preliminary_fever_500.json`

本地数据准备完成后即可跑 FEVER **C0** 和 **C3**。正式 FEVER **C1** 必须等 integrator 的全库检索 / index；**不能**使用早先 100k smoke-test candidate pool。

## 2. 创建并激活 `.venv`

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

若 PowerShell 阻止激活脚本，先执行一次：

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## 3. 安装 PyTorch，检查 CUDA

NVIDIA CUDA 13.0：

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu130
```

CPU 或其他 CUDA 版本：用 [PyTorch 官网](https://pytorch.org/get-started/locally/) 上对应本机的安装命令。

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

`True` 表示 PyTorch 能看到 GPU。`False` 表示在用 CPU，或 CUDA 未装对。4-bit NF4 加载 Qwen 需要 CUDA。

## 4. 安装 `requirements.txt`

```powershell
pip install -r requirements.txt
```

## 5. 下载数据

```powershell
python scripts/download_data.py
```

会拉取 Natural Questions validation（Hugging Face `datasets`，保存到 `data/raw/natural_questions/dev`）、FEVER `train.jsonl` / `shared_task_dev.jsonl`，以及 FEVER Wikipedia `wiki-pages.zip`（解压到 `data/raw/fever/wiki-pages/`）。已存在的文件会跳过。

## 6. 验证数据

```powershell
python scripts/verify_data.py
```

只读检查，不修改数据。全部通过时打印 `DATA VALIDATION PASSED`。

## 7. NQ preprocessing / NQ 预处理

```powershell
python scripts/prepare_nq.py
```

从 `data/raw/natural_questions/dev` 生成 controlled subset：`data/processed/nq_controlled/dev.jsonl` 和 `stats.json`。不修改原始数据，不跑检索。

## 8. retrieval-data preprocessing / 检索数据预处理

```powershell
python scripts/prepare_retrieval_data.py
```

写出 Retriever 用的 JSONL（不做 embedding、不建 FAISS index、不跑语言模型）：

- `data/processed/retrieval/nq_dev.jsonl`
- `data/processed/retrieval/nq_corpus.jsonl`
- `data/processed/retrieval/fever_dev.jsonl`
- `data/processed/retrieval/fever_corpus.jsonl`

NQ 的正式 FAISS index 需要在本地用已验证过的脚本生成（同样不在 GitHub 上）：

```powershell
python scripts/run_nq_bge_retriever.py
```

这会编码 NQ corpus 并写入 `data/processed/indexes/nq_bge_base.index` 与 `nq_bge_base_meta.jsonl`。E5 对照文件 `nq_e5_base.index` / `nq_e5_base_meta.jsonl` 由 `python scripts/run_nq_retriever.py` 生成，仅作历史对比。正式 NQ C1 使用 **BGE + IndexFlatIP**：`nq_bge_base.index` + `nq_bge_base_meta.jsonl`。

FEVER 正式 C1 **不是** `scripts/run_fever_base_rag.py` 里那个 100k controlled smoke-test pool；那次检索不能当作正式实验结果。全库 FEVER index 由 integrator 构建，组员不要自行运行 `python scripts/build_fever_index.py`，除非之后另有明确要求。

## 哪些可直接复用，哪些必须本地重新生成

clone 后仓库里有代码、`configs/`、`docs/`，以及（若已提交）`results/` 下的小型 json/jsonl。Preliminary Recoverability Experiment 必须使用冻结的 500-ID manifests，不要重新抽样。共用 `data/` 按上面的脚本在本地重建。

需要在本机重新生成的大文件：

| 内容 | 原因 |
|---|---|
| `data/raw/` | `.gitignore`，含 NQ disk dataset、FEVER jsonl、wiki-pages |
| `data/processed/nq_controlled/` | `.gitignore` |
| `data/processed/retrieval/` | `.gitignore`；FEVER corpus 约数百万页 |
| `data/processed/indexes/` | `.gitignore`；NQ BGE/E5 FlatIP、FEVER IVFPQ index、embedding shards 与 passage ids |
| `Qwen/Qwen2.5-7B-Instruct` | Hugging Face cache，不在仓库内；第一次跑 generator 时自动下载 |
| `BAAI/bge-base-en-v1.5` | 同样由 `sentence-transformers` 下载到 HF cache |

本机如果已经跑过上述脚本，对应目录非空即可跳过，脚本对已存在的下载也会跳过。不要覆盖 `results/` 里已有的 3B smoke-test 文件。

## Directory / 目录

- `configs/` — 正式配置。generator 与 retrieval 以 `configs/base.yaml` 为准。
- `data/raw/` — 官方原始数据（不入库）。
- `data/processed/` — NQ subset、retrieval JSONL、FAISS index（不入库）。
- `src/` — 检索、生成、评测代码。
- `scripts/` — 下载、验证、预处理、smoke test。
- `results/` — 检索指标与 smoke-test 输出。
- `docs/` — 环境与冻结检索说明。
