# Baseline evaluation

Two formal baselines on the same frozen NQ Test (before Controller / Enhancement), plus the Adaptive Product evaluation harness on `feature/product-evaluation`:

| System | Definition |
|----------|------------|
| **Qwen2.5-7B only** | `Qwen/Qwen2.5-7B-Instruct`, closed-book (no retrieval) |
| **Qwen2.5-7B + matched RAG** | Same Qwen + NQ-Train matched BGE FAISS Top-5 + `context_fallback` |
| **Adaptive Product** | Production `load_current_pipeline` → Reference RAG → Diagnosis → P60 → Controller → Enhancement → `final_answer` |

Same frozen Test IDs, same gold answers, same automatic EM / Token F1 / Normalized Accuracy (`baselines/_lib/answer_scoring.py`). Product AIC@5 uses **initial** Reference RAG Top-5 only.

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
python baselines/judge/run_llm_judge.py --input adaptive_product --live --output-dir baselines/outputs/judge/adaptive_product_full
```

## Adaptive Product evaluation

Uses production entry only (`src.pipeline.current.load_current_pipeline`). Does **not** call `run_baseline_evaluation.py` for product answers. Sealed Test is locked unless `--allow-sealed-test` is passed.

**Train/Val engineering smoke** (not formal Test):

```bash
python baselines/run_product_evaluation.py --smoke --eval-split train --limit 3
```

**Sealed Test dry validation** (IDs/assets only; no inference):

```bash
python baselines/run_product_evaluation.py --validate-split-only --eval-split test --allow-sealed-test
```

**Formal Test** (11114; do not run until harness is ready and GPU budget allocated):

```bash
python baselines/run_product_evaluation.py --eval-split test --allow-sealed-test
```

Formal product outputs: `baselines/outputs/adaptive_product_test/`.

## Adaptive Product Test (formal, n=11114)

Automatic metrics only (LLM Judge **not** run yet for this product arm).
Source of truth: `baselines/outputs/adaptive_product_test/metrics.json`.

| Metric | Value |
|--|--:|
| **n** | 11114 |
| **EM** | 0.29575310419290984 |
| **Token F1** | 0.448427960309939 |
| **Normalized Accuracy** | 0.2985423789814648 |
| **AIC@5** (initial evidence) | 0.6997480655029692 |
| **Final AIC@5** | 0.7029872233219363 |
| **selected_rate** | 0.054885729710275326 |
| **selected_n** | 610 |
| **no_action** | 10504 |
| **retrieval_enhancement** | 403 |
| **evidence_refinement** | 117 |
| **generation_repair** | 90 |

Formal artifacts:
- `baselines/outputs/adaptive_product_test/predictions.jsonl`
- `baselines/outputs/adaptive_product_test/metrics.json`
- `baselines/outputs/adaptive_product_test/run_config.json`

## Outputs

| Artifact | Directory |
|----------|-----------|
| Qwen-only baseline | `baselines/outputs/qwen_only/` |
| Matched RAG baseline | `baselines/outputs/rag_matched_test_context_fallback/` |
| Adaptive Product | `baselines/outputs/adaptive_product_test/` |
| Qwen-only full Judge | `baselines/outputs/judge/qwen_only_full/` |
| Matched RAG full Judge | `baselines/outputs/judge/rag_matched_full/` |
| Adaptive Product full Judge | `baselines/outputs/judge/adaptive_product_full/` |

Each baseline/product run writes `predictions.jsonl`, `metrics.json`, `run_config.json`.
Each full Judge run writes `judge_results.jsonl`, `judge_summary.json`, `run_config.json`.

Temporary pilot/smoke/retry/dryrun dirs, FAISS index/corpus, `_backup/`, and secrets must **not** be committed.

## Automatic metrics

- **EM** — exact match after standard answer normalization (`src.evaluation.qa.best_em_f1`)
- **Token F1** — token-level F1 (same backend)
- **Normalized Accuracy** — form-only match (Unicode / case / punctuation / articles / whitespace / separators). **Not** semantic similarity and **not** an LLM Judge.
- **Judge-assisted Accuracy** — `auto_correct` or LLM `CORRECT`; `UNCERTAIN` ≠ correct.
