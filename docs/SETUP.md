# Setup

Windows 环境。PyTorch **不在** `requirements.txt` 里，需按本机硬件单独安装，再装其余依赖。

模型和原始/预处理数据都很大，**不会提交到 GitHub**（见 `.gitignore`：`.venv/`、`data/raw/*`、`data/processed/*`、`models/`、`checkpoints/`）。clone 之后需要在本地重新下载数据、跑预处理，并在需要跑 Reference RAG 时本地生成 FAISS index。`results/` 里体积较小的 smoke-test / retrieval metrics 可以随仓库保留，用于对照，不是可执行的数据或模型。

当前正式 Retriever：**BGE + FAISS + Top-5**（`BAAI/bge-base-en-v1.5`）。Primary SLM：`Qwen/Qwen2.5-7B-Instruct`，4-bit NF4。第一次加载该模型时，Hugging Face 会把它下载到本地 cache（不在本仓库里）。

**smoke test** = 工程验证（工程能否跑通）。  
**Preliminary Recoverability Experiment** = 尚未正式开始。

## 1. Clone repository / 克隆存储库

```powershell
git clone <repository-url>
cd capability-aware-slm-rag
```

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

这会编码 NQ corpus 并写入 `data/processed/indexes/nq_bge_base.index` 与 `nq_bge_base_meta.jsonl`。E5 对照 index 由 `python scripts/run_nq_retriever.py` 生成，仅作历史对比，正式 Retriever 仍是 BGE + FAISS + Top-5。

## 哪些可直接复用，哪些必须本地重新生成

clone 后仓库里有代码、`configs/`、`docs/`，以及（若已提交）`results/` 下的小型 json/jsonl。

需要在本机重新生成的大文件：

| 内容 | 原因 |
|---|---|
| `data/raw/` | `.gitignore`，含 NQ disk dataset、FEVER jsonl、wiki-pages |
| `data/processed/nq_controlled/` | `.gitignore` |
| `data/processed/retrieval/` | `.gitignore`；FEVER corpus 约数百万页 |
| `data/processed/indexes/` | `.gitignore`；BGE/E5 FAISS index 与 passage metadata |
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
