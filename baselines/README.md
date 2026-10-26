# Baseline evaluation

Independent baselines for fair comparison **before** Controller / Enhancement.
No Diagnosis, Probe, Controller, or Enhancement code is used here.

## Baselines

| Mode | Definition |
|------|------------|
| `qwen_only` | Qwen2.5-7B-Instruct, **no** retrieved evidence (closed-book). |
| `rag` | Same Qwen + BGE-base + FAISS **Top-5** evidence → RAG prompt. |

Shared across modes: same generator config (`configs/base.yaml`), same Test IDs, same gold, same EM/F1 (`src.evaluation.qa.best_em_f1`). The only intended variable is Top-5 RAG evidence.

## Frozen Test

- Formal Test IDs come **only** from the original  
  `diagnosis_nq_train_74097_split.json` (`splits.nq.test`, **n=11114**).
- Place that file at `baselines/splits/diagnosis_nq_train_74097_split.json`  
  (or pass `--split-file`).
- **Do not** regenerate or guess IDs. If the file is missing, the runner exits with `FROZEN_SPLIT_MISSING`.

## Commands

Smoke (synthetic, no frozen Test; dry-run avoids GPU):

```bash
python baselines/run_baseline_evaluation.py --mode qwen_only --smoke --dry-run
python baselines/run_baseline_evaluation.py --mode rag --smoke --dry-run
```

Formal (after frozen split is in place; **not** run automatically):

```bash
python baselines/run_baseline_evaluation.py --mode qwen_only --split-file baselines/splits/diagnosis_nq_train_74097_split.json
python baselines/run_baseline_evaluation.py --mode rag --split-file baselines/splits/diagnosis_nq_train_74097_split.json
```

Resume:

```bash
python baselines/run_baseline_evaluation.py --mode rag --split-file baselines/splits/diagnosis_nq_train_74097_split.json --resume
```

Validate split only:

```bash
python baselines/run_baseline_evaluation.py --mode rag --validate-split-only --split-file baselines/splits/diagnosis_nq_train_74097_split.json
```

## Outputs

Under `baselines/outputs/<mode>/`:

- `predictions.jsonl` — per-sample predictions (+ RAG passage ids/ranks)
- `metrics.json` — aggregate EM / F1
- `run_config.json` — frozen run settings

Large outputs are gitignored via `baselines/.gitignore`.
