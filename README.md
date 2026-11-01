# Capability-Aware Adaptive Enhancement for Small-Model RAG

## Overview

This repository is a capability-aware retrieval-augmented generation system built on Qwen2.5-7B-Instruct. A reference RAG pass produces an initial answer. A hierarchical diagnosis model then labels that answer, and a frozen P60 selector decides whether enhancement is warranted. A controller routes selected failures to one enhancement action. The final runtime leaves the evidence probe disabled.

The formal comparison is three systems on one frozen NQ Test of 11,114 examples: Qwen2.5-7B only, matched RAG, and the adaptive product.

## System Architecture

```text
Query
→ Reference RAG
→ Initial Answer
→ Diagnosis
→ P60 Selection
→ Controller
→ Optional Enhancement
→ Final Answer
```

Reference RAG uses Qwen2.5-7B-Instruct (NF4, greedy decoding, 64 new tokens) and BGE Top-5 retrieval with the `context_fallback` prompt.

Diagnosis is the hierarchical classifier in `models/diagnosis/diagnosis_v2_expanded_hierarchical.json`. Its stage-1 failure threshold is `0.424`.

P60 selection reads `configs/controller/p60_online_adapter.json`: a 78-dimensional feature vector, frozen z-statistics, `t_insuff` `0.5349365680553068`, and `t_combo` `-3.0`. A correct diagnosis is never enhanced. A failure is selected when insufficiency probability meets `t_insuff`, or otherwise when the frozen combo score meets `t_combo`.

The controller maps the diagnosis label to one action. The probe is off (`probe_enabled: false` in `configs/controller/current_baseline.json`).

| Diagnosis label | Controller action |
|---|---|
| Correct | `no_action` |
| Evidence Insufficiency | `retrieval_enhancement` |
| Evidence Accessibility Failure | `evidence_refinement` |
| Evidence Utilisation Failure | `generation_repair` |

## Diagnostic Formulation

The diagnosis model assigns one label:

- **Correct.** The initial answer is treated as acceptable, and the controller keeps it.
- **Evidence Insufficiency.** The retrieved passages do not contain enough evidence to answer.
- **Evidence Accessibility Failure.** Supporting evidence is present, but it is not ranked where the generator can use it.
- **Evidence Utilisation Failure.** Usable evidence is accessible, but the generator does not turn it into the answer.

## Repository Structure

| Path | Role |
|---|---|
| `src/` | Reference RAG, diagnosis, P60 selection, controller, and enhancement |
| `configs/` | Production registration, RAG, resources, and P60 thresholds |
| `models/` | Diagnosis classifier and P60 histogram-gradient-boosting models |
| `baselines/` | Baseline and product evaluation runners, judge harness, frozen split, and formal outputs |
| `tests/` | CPU tests for the pipeline, scoring, and evaluation harnesses |
| `baselines/outputs/qwen_only/` | Formal Qwen-only Test |
| `baselines/outputs/rag_matched_test_context_fallback/` | Formal matched RAG Test |
| `baselines/outputs/adaptive_product_test/` | Formal adaptive product Test |
| `baselines/outputs/judge/` | Formal LLM Judge results for the three systems |

## Environment

The checked runtime is Python 3.12.10 with the packages below. `requirements.txt` pins older library versions; install a CUDA PyTorch build first, then `requirements-controller.txt`, and install `xgrammar==0.2.8` before generation repair.

| Package | Checked version | Requirements pin |
|---|---|---|
| PyTorch | 2.14.0+cu130 | install separately (CUDA) |
| transformers | 5.17.0 | 4.57.6 |
| sentence-transformers | 6.0.1 | 5.1.2 |
| faiss | 1.15.1 | `faiss-cpu` |
| scikit-learn | 1.9.1 | 1.9.1 in `requirements-controller.txt` |
| bitsandbytes | 0.50.2 | unpinned |
| xgrammar | 0.2.8 | required by `src/compensation/qwen_constrained.py` |

Importing `triton` in this environment reports 3.8.0. Triton is not pinned in `requirements.txt`. P60 uses scikit-learn histogram gradient boosting. The production reranker loads on CPU.

## Required External Models and Data

Already in the repository:

| Path | Role |
|---|---|
| `configs/controller/current_baseline.json` | Production registration |
| `configs/controller/reference_rag_v1.json` | Generator, index paths, and prompt |
| `configs/controller/nq_train_resources_v1.json` | Expected corpus and index sizes and checksums |
| `configs/controller/p60_online_adapter.json` | P60 feature names, thresholds, and z-statistics |
| `models/diagnosis/diagnosis_v2_expanded_hierarchical.json` | Diagnosis classifier |
| `models/controller/p60_online_adapter/*.joblib` | P60 models |
| `artifacts/controller/p60_online_adapter/training_provenance.json` | Provenance checked at startup |
| `baselines/splits/diagnosis_nq_train_74097_split.json` | Frozen split, including the 11,114 Test ids |
| `baselines/outputs/` | Formal Test and Judge results listed above |

Prepare these separately. They are too large for Git. Paths come from `configs/controller/nq_train_resources_v1.json` and `configs/controller/reference_rag_v1.json`.

| Resource | Where the configs expect it |
|---|---|
| `Qwen/Qwen2.5-7B-Instruct` | Hugging Face cache. NF4, `do_sample: false`, `max_new_tokens: 64` |
| `BAAI/bge-base-en-v1.5` | Hugging Face cache. Query prefix: `Represent this sentence for searching relevant passages: ` |
| `BAAI/bge-reranker-large` | Hugging Face cache, revision `55611d7bca2a7133960a6d3b71e083071bbfc312`, CPU FP32 (`src/compensation/retrieval_enhancement/p3.py`) |
| NQ FAISS index | `data/processed/indexes/nq_train_bge_base.index` (3,760,493,613 bytes) |
| Passage ids | `data/processed/indexes/nq_train_bge_base_ids.jsonl` (37,947,689 bytes) |
| Offsets | `data/processed/indexes/nq_train_bge_base_offsets.npy` (9,793,080 bytes) |
| Corpus | `data/processed/retrieval/nq_train_corpus.jsonl` (782,900,757 bytes) |

`expected_ntotal` is 1,224,119 passages. Checksums are in `configs/controller/nq_train_resources_v1.json`.

## Evaluation Dataset

The formal comparison uses the frozen NQ Test list in `baselines/splits/diagnosis_nq_train_74097_split.json` (`splits.nq.test`). It contains 11,114 ids. SHA-256 recorded by the product run: `e6572c9b1c8f92dccceef2a85200a891ab33c01fe4775e4541c55e86f64de9db`.

Qwen-only, matched RAG, and the adaptive product are scored on these same ids, in the same order.

## Compared Systems

1. **Qwen2.5-7B only.** Closed-book generation. No retrieval and no controller. Config: `baselines/configs/qwen_only.json`.
2. **Matched RAG.** The same generator, NQ-Train BGE Top-5, and the `context_fallback` prompt. No diagnosis or enhancement. Config: `baselines/configs/rag_matched.json`.
3. **Adaptive product.** Matched reference RAG, then diagnosis, P60 selection, and the controller. A failure label can call retrieval enhancement, evidence refinement, or generation repair. Config: `baselines/configs/adaptive_product.json`, which points at `configs/controller/current_baseline.json`.

## Evaluation Metrics

Automatic scores come from `baselines/_lib/answer_scoring.py`. EM and token F1 use `src.evaluation.qa.best_em_f1`.

- **Exact Match (EM).** Exact match after the standard answer normalization.
- **Token F1.** Mean token-level F1. `f1` and `mean_f1` in the metrics files are the same mean.
- **Normalized Accuracy.** A form-only match after Unicode, case, punctuation, article, whitespace, and separator normalization. It is not a semantic score.
- **Answer-in-Context@5 (AIC@5).** Whether any gold string appears in the Top-5 passages. Qwen-only has no retrieved context, so the cell is `—`. For RAG and the product, `answer_in_context_at_5` uses the initial evidence.
- **Judge-assisted Accuracy.** A row counts as correct when the automatic normalized match is already correct, or when the LLM judge returns `CORRECT`. `UNCERTAIN` is not counted as correct. The judge does not replace EM or token F1.

The product metrics also report the selected rate, the four action counts, and final AIC@5 on the evidence kept with the final answer.

## Final Results

The cells below are read from the formal JSON files and shown as percentages to two decimal places. `n` is 11,114 in every file.

| Metric | Qwen-only | Matched RAG | Adaptive Product |
|---|---:|---:|---:|
| Exact Match | 10.72% | 29.09% | 34.07% |
| Token F1 | 20.68% | 44.34% | 49.55% |
| Normalized Accuracy | 10.78% | 29.40% | 34.42% |
| AIC@5 | — | 69.97% | 69.97% |
| Judge-assisted Accuracy | 29.55% | 57.50% | 62.91% |

Sources:

- Qwen-only: `baselines/outputs/qwen_only/metrics.json`, `baselines/outputs/judge/qwen_only_full/judge_summary.json`
- Matched RAG: `baselines/outputs/rag_matched_test_context_fallback/metrics.json`, `baselines/outputs/judge/rag_matched_full/judge_summary.json`
- Adaptive product: `baselines/outputs/adaptive_product_test/metrics.json`, `baselines/outputs/judge/adaptive_product_full/judge_summary.json`

Product-specific fields from `baselines/outputs/adaptive_product_test/metrics.json`:

| Field | Value |
|---|---:|
| selected rate | 50.51% |
| selected count | 5614 |
| `no_action` | 5500 |
| `retrieval_enhancement` | 2049 |
| `evidence_refinement` | 1990 |
| `generation_repair` | 1575 |
| initial AIC@5 | 69.97% (7777 / 3337) |
| final AIC@5 | 74.10% (8236 / 2878) |

## Reproduction

### 1. Environment setup

Create a Python 3.12 environment.

### 2. Install dependencies

Install a CUDA PyTorch build, then:

```bash
pip install -r requirements-controller.txt
pip install xgrammar==0.2.8
```

### 3. Prepare external resources

Download `Qwen/Qwen2.5-7B-Instruct`, `BAAI/bge-base-en-v1.5`, and `BAAI/bge-reranker-large` at the revision above. Place the NQ index, ids, offsets, and corpus at the paths in `configs/controller/nq_train_resources_v1.json`.

### 4. Validate configuration/resources

```bash
python scripts/check_controller_current.py
```

`ready_for_gpu` is true only when the registered classifier, P60 models, and provenance files are present. This command does not read the frozen Test.

### 5. Run tests

```bash
python -m pytest tests/test_controller_current.py tests/test_controller.py tests/test_controller_routing.py tests/test_controller_selection.py tests/test_controller2_module_integration.py tests/test_adaptive_rag_pipeline.py tests/test_diagnosis_classifier.py tests/test_diagnosis_hierarchical_classifier.py tests/test_diagnosis_schema.py tests/test_diagnosis_features.py tests/test_p60_online_feature_provider.py tests/test_enhancement_interfaces.py tests/test_retrieval_enhancement.py tests/test_evidence_refinement.py tests/test_baseline_evaluation.py tests/test_product_evaluation.py tests/test_answer_scoring.py tests/test_llm_judge.py tests/test_compensation_schema.py tests/test_reference_rag.py tests/test_team_nq_boundary.py -q
```

These tests do not execute the 11,114-row Test and do not call the judge API.

### 6. Qwen-only evaluation

```bash
python baselines/run_baseline_evaluation.py --mode qwen_only --split-file baselines/splits/diagnosis_nq_train_74097_split.json
```

This writes `baselines/outputs/qwen_only/` and needs the generator.

### 7. Matched RAG evaluation

```bash
python baselines/run_baseline_evaluation.py --mode rag --config baselines/configs/rag_matched.json --eval-split test --rag-prompt context_fallback --split-file baselines/splits/diagnosis_nq_train_74097_split.json
```

The default output directory for this config, split, and prompt is `baselines/outputs/rag_matched_test_context_fallback/`.

### 8. Adaptive Product evaluation

```bash
python baselines/run_product_evaluation.py --config baselines/configs/adaptive_product.json --eval-split test --allow-sealed-test --split-file baselines/splits/diagnosis_nq_train_74097_split.json
```

`--allow-sealed-test` is required for the frozen Test. The default output directory is `baselines/outputs/adaptive_product_test/`. The default feature batch size is 8. Qwen decoding stays serial.

### 9. LLM Judge

The judge calls Azure OpenAI only when `--live` is set. Put credentials in the environment variables named by `baselines/judge/judge_config.json`: `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_BASE_URL`, and `AZURE_OPENAI_DEPLOYMENT`.

```bash
python baselines/judge/run_llm_judge.py --input qwen_only --live --output-dir baselines/outputs/judge/qwen_only_full
python baselines/judge/run_llm_judge.py --input rag_matched_test_context_fallback --live --output-dir baselines/outputs/judge/rag_matched_full
python baselines/judge/run_llm_judge.py --input baselines/outputs/adaptive_product_test/predictions.jsonl --live --output-dir baselines/outputs/judge/adaptive_product_full
```

Without `--live`, the judge does not call the API.

## Evaluation Outputs

### `predictions.jsonl`

One JSON record per evaluation sample. Rows include `sample_id`, `query`, the prediction, gold answers, and the automatic sample scores. RAG and product rows also include retrieved passage ids. Product rows add the initial answer, diagnosis label, whether enhancement was selected, the controller action, and the enhancement status.

### `metrics.json`

Aggregate deterministic metrics for that directory: EM, token F1, normalized accuracy, and, where retrieval exists, AIC@5. Product files also include the selected rate, action counts, and final AIC@5.

### `run_config.json`

The config path, split, model, retrieval settings, and other reproducibility fields recorded for that run.

### `judge_results.jsonl`

One semantic judge decision per sample.

### `judge_summary.json`

Aggregate judge counts, including `judge_assisted_accuracy`, and the API call counts for that run.

Formal directories:

- `baselines/outputs/qwen_only/`
- `baselines/outputs/rag_matched_test_context_fallback/`
- `baselines/outputs/adaptive_product_test/`
- `baselines/outputs/judge/qwen_only_full/`
- `baselines/outputs/judge/rag_matched_full/`
- `baselines/outputs/judge/adaptive_product_full/`
