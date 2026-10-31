# Baseline evaluation

Two formal baselines on the same frozen NQ Test (before Controller / Enhancement):

| Baseline | Definition |
|----------|------------|
| **Qwen2.5-7B only** | `Qwen/Qwen2.5-7B-Instruct`, closed-book (no retrieval) |
| **Qwen2.5-7B + matched RAG** | Same Qwen + NQ-Train matched BGE FAISS Top-5 + `context_fallback` |

Same generator (`configs/base.yaml`), same Test IDs, same gold answers, same automatic EM / Token F1 / Normalized Accuracy.

## Frozen Test

- Split file: `baselines/splits/diagnosis_nq_train_74097_split.json`
- IDs: `splits.nq.test` only, **n=11114**
- **Do not re-split or regenerate Test IDs.** Missing file → `FROZEN_SPLIT_MISSING`.

## Frozen Test Results

Source of truth: `metrics.json` / `judge_summary.json` under the formal output dirs below.
LLM Judge rubric was calibrated before full evaluation; temporary pilot/smoke outputs are **not** formal results.

| | Qwen2.5-7B only | Qwen2.5-7B + BGE RAG |
|--|----------------:|--------------------:|
| **Config** | closed-book | `BAAI/bge-base-en-v1.5`, matched NQ-Train FAISS, Top-k=5, `context_fallback` |
| **n** | 11114 | 11114 |
| **EM** | 0.10716213784416052 | 0.29089436746445924 |
| **Token F1** | 0.20683020705954444 | 0.44335644383133416 |
| **Normalized Accuracy** | 0.10779197408673745 | 0.2939535720712615 |
| **AIC@5** | — | 0.6997480655029692 |
| **Judge-assisted Accuracy** | 0.2954831743746626 | 0.5750404894727371 |

- **Judge-assisted delta (RAG − Qwen-only):** +0.2795573150980745 (**+27.96 pp**).
- **UNCERTAIN** is counted conservatively as **not correct**.
- LLM Judge model: Azure **GPT-6.1-sol** (Responses API; secrets via env vars only).
- Judge rubric was **frozen before** the formal full evaluation.
- Formal outputs:
  - `baselines/outputs/qwen_only/`
  - `baselines/outputs/rag_matched_test_context_fallback/`
  - `baselines/outputs/judge/qwen_only_full/`
  - `baselines/outputs/judge/rag_matched_full/`

**Invalidated (do not cite):** an earlier RAG Test used Dev index `nq_bge_base` (~99k) instead of the NQ-Train matched corpus/index. Reported ~EM 8.3% / F1 15% / normalized 8.4% is **void** due to retrieval corpus/index mismatch with `diagnosis_nq_train_74097`.

## Matched RAG freeze

- Generator: `Qwen/Qwen2.5-7B-Instruct`, NF4, greedy (`do_sample=false`), `max_new_tokens=64`
- Retriever: `BAAI/bge-base-en-v1.5`, FAISS IndexFlatIP, Top-k=5
- Index: `data/processed/indexes/nq_train_bge_base.index`
- Ids / offsets / corpus:
  - `data/processed/indexes/nq_train_bge_base_ids.jsonl`
  - `data/processed/indexes/nq_train_bge_base_offsets.npy`
  - `data/processed/retrieval/nq_train_corpus.jsonl`
- Prompt: `context_fallback`
- Config: `baselines/configs/rag_matched.json`

## Formal run commands

**Qwen-only**

```bash
python baselines/run_baseline_evaluation.py --mode qwen_only --split-file baselines/splits/diagnosis_nq_train_74097_split.json
```

**Matched RAG**

```bash
python baselines/run_baseline_evaluation.py --mode rag --config baselines/configs/rag_matched.json --eval-split test --rag-prompt context_fallback --split-file baselines/splits/diagnosis_nq_train_74097_split.json
```

**Matched RAG resume**

```bash
python baselines/run_baseline_evaluation.py --mode rag --config baselines/configs/rag_matched.json --eval-split test --rag-prompt context_fallback --split-file baselines/splits/diagnosis_nq_train_74097_split.json --resume
```

**Full LLM Judge** (requires Azure env vars; incurs cost)

```bash
python baselines/judge/run_llm_judge.py --input qwen_only --live --output-dir baselines/outputs/judge/qwen_only_full
python baselines/judge/run_llm_judge.py --input rag_matched_test_context_fallback --live --output-dir baselines/outputs/judge/rag_matched_full
```

## Outputs

| Artifact | Directory |
|----------|-----------|
| Qwen-only baseline | `baselines/outputs/qwen_only/` |
| Matched RAG baseline | `baselines/outputs/rag_matched_test_context_fallback/` |
| Qwen-only full Judge | `baselines/outputs/judge/qwen_only_full/` |
| Matched RAG full Judge | `baselines/outputs/judge/rag_matched_full/` |

Each baseline run writes `predictions.jsonl`, `metrics.json`, `run_config.json`.
Each full Judge run writes `judge_results.jsonl`, `judge_summary.json`, `run_config.json`.

Temporary pilot/smoke/retry/dryrun dirs, FAISS index/corpus, `_backup/`, and secrets must **not** be committed.

## FEVER Qwen2.5-7B only (formal)

Frozen FEVER Test (independent of NQ outputs):

| | Value |
|--|------:|
| **Dataset** | FEVER `fever_dev` (SUPPORTS / REFUTES only) |
| **n** | 13332 |
| **SUPPORTS / REFUTES** | 6666 / 6666 |
| **Accuracy** | 0.7216471647164716 |
| **Macro-F1** | 0.7059210905831337 |
| **SUPPORTS F1** | 0.6379158942335837 |
| **REFUTES F1** | 0.7739262869326835 |
| **invalid_n** | 0 |

- Bundle: `baselines/data/fever_dev_13332.jsonl` + `baselines/splits/fever_dev_13332_split.json`
- Config: `baselines/configs/fever_qwen_only.json` (closed-book; no retrieval)
- Formal output: `baselines/outputs/fever_qwen_only/`

```bash
python baselines/run_baseline_evaluation.py --mode fever_qwen_only
```

## Automatic metrics

- **EM** — exact match after standard answer normalization (`src.evaluation.qa.best_em_f1`)
- **Token F1** — token-level F1 (same backend)
- **Normalized Accuracy** — form-only match (Unicode / case / punctuation / articles / whitespace / separators). **Not** semantic similarity and **not** an LLM Judge.
- **Judge-assisted Accuracy** — `auto_correct` or LLM `CORRECT`; `UNCERTAIN` ≠ correct.
