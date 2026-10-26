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

## Formal Test metrics (n=11114)

Automatic metrics only; **LLM Judge is not included**.

**Qwen-only**

| Metric | Value |
|--------|------:|
| EM | 0.1072 |
| Token F1 | 0.2068 |
| Normalized Accuracy | 0.1078 |

**Matched RAG** (`context_fallback`)

| Metric | Value |
|--------|------:|
| EM | 0.2909 |
| Token F1 | 0.4434 |
| Normalized Accuracy | 0.2940 |
| answer-in-context@5 | 0.6997 |

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

## Outputs

| Baseline | Directory |
|----------|-----------|
| Qwen-only | `baselines/outputs/qwen_only/` |
| Matched RAG | `baselines/outputs/rag_matched_test_context_fallback/` |

Each run writes `predictions.jsonl`, `metrics.json`, `run_config.json`.

Large artifacts (`predictions.jsonl`, FAISS index, corpus, `_backup/`) are gitignored by default via `baselines/.gitignore` and should **not** be committed.

## Automatic metrics

- **EM** — exact match after standard answer normalization (`src.evaluation.qa.best_em_f1`)
- **Token F1** — token-level F1 (same backend)
- **Normalized Accuracy** — form-only match (Unicode / case / punctuation / articles / whitespace / separators). **Not** semantic similarity and **not** an LLM Judge.
