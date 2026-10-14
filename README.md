# capability-aware-slm-rag

本项目是一个针对 Small-Model RAG 的 diagnostic and adaptive enhancement layer。当前 repository 中的 Reference RAG 用于开发和实验验证，不代表项目创新点本身。

## Reference RAG

- Primary SLM: Qwen2.5-7B-Instruct
- Retriever embedding: BAAI/bge-base-en-v1.5
- Vector search: FAISS
- Top-k: 5
- Quantization: 4-bit NF4
- Generation: deterministic (`do_sample=false`)

## Data

- Natural Questions validation: 7830
- NQ controlled subset: 3216
- NQ retrieval corpus: 99,922 passages
- FEVER train: 145,449
- FEVER dev: 19,998
- processed FEVER dev: 6666 SUPPORTS + 6666 REFUTES
- FEVER valid gold-evidence samples: 13,229 / 13,332
- FEVER retrieval corpus: 5,396,106 passages

## Status

- data preparation completed
- retrieval pilot completed
- Reference RAG engineering validation completed
- primary SLM changed from 3B to 7B
- NQ and FEVER 7B C0/C1/C3 runs are complete across the experiment branches; frozen results and judge re-analysis are documented in [project status and direction](docs/PROJECT_STATUS_AND_DIRECTION.md).
- Diagnosis v2-expanded, Probe, Safe Binary and Controller are frozen on `feature/diagnosis-probe-zheng` at `7c7863f`; project-sample annotations are published there. This E branch consumes the shared decision contract.
- `reranking` at `338ff34` implements evidence refinement with offline retrieval evaluation; generator-level gains have not been measured there.
- E implements Verification / Answer Repair with a Controller adapter, resumable experiments, provenance checks and paired evaluation. v4–v6r1 real T4 runs are archived. v6r1 completed 128 predictions (123 unique IDs), 81 cloud tests and frozen-code replay of all results. FEVER reviewed accuracy improves 4/14→7/14, but dev50 falls 45→44; NQ dev50 EM falls 20→7 with longer extracted answers and some irrelevant quotes. Neither new strategy is promoted. Human correctness/support review is deferred; the 42 changed pairs are prioritized. See [current handoff](docs/ANSWER_REPAIR_HANDOFF.md), [latest results](docs/ANSWER_REPAIR_RESULTS_20261014.md), and [Colab / Kaggle](docs/ANSWER_REPAIR_COLAB.md). Engineering and planned development experiments are complete; reliable repair gains and held-out acceptance remain unverified. The delivery branch is `feature/verification-answer-repair`; code, frozen archives, results and review records are versioned together.
- The frozen preliminary protocol remains in [docs/PRELIMINARY_EXPERIMENT.md](docs/PRELIMINARY_EXPERIMENT.md).

2026-10-14 (later): a second, blind read of the 28 key v6r1 cases agrees with the assisted review on all 9 fixes and 3 regressions, and the user confirmed the sheet (9 fixes / 3 regressions / 2 deferred after de-duplication; 97 unique samples still pending batch confirmation); the confirmation sheet is [docs/ANSWER_REPAIR_CONFIRMATION_20261014.md](docs/ANSWER_REPAIR_CONFIRMATION_20261014.md). E v7 is frozen before any GPU run ([docs/ANSWER_REPAIR_V7.md](docs/ANSWER_REPAIR_V7.md)): NQ returns a short answer inside a located quote; FEVER lets only windows naming the claim's subject vote. Replaying v6r1's stored NLI scores under the v7 rule gives FEVER reviewed 4→10 and dev50 45→46 with no regression (design-stage evidence, not held-out). The v7 bundle's E-untouched held-out cohort (FEVER 267, NQ 146) was run once on Kaggle ([results](docs/ANSWER_REPAIR_RESULTS_20261014_V7.md)): FEVER 1 fix / 4 regressions, NQ frozen EM 71→62 with real semantic regressions; neither v7 strategy is promoted. The NQ 3216-pool held-out (120 utilisation labels) is frozen as an ID list and waits for gold answers.

Agent-assisted semantic review now covers all 128 v6r1 results: 123 unique cases, provisional fixes/regressions and citation judgments with evidence anchors. Human confirmation remains pending; agent suggestions are not human gold. See the assisted-review section of the [latest results](docs/ANSWER_REPAIR_RESULTS_20261014.md), the [128-case review](artifacts/answer_repair/v6r1_reviewed/assisted_review_20261014/assisted_review.md), and the [archive inventory](artifacts/answer_repair/git_archive_manifest.json).

## Engineering Validation

These are **engineering smoke tests** only. They do **not** answer the Preliminary Recoverability Experiment research questions.

**NQ 3B historical smoke test v2** (`results/base_rag/nq_smoke_test_v2.jsonl`): 20 samples; EM = 0.3000; Average F1 = 0.3460; gold evidence in Top-5 = 17/20; retrieval miss = 3/20; gold in Top-5 but wrong = 11/20.

**NQ 7B smoke test** (`results/base_rag/nq_smoke_test_7b.jsonl`): 20 samples; EM = 0.4000; Average F1 = 0.4770; gold evidence in Top-5 = 17/20; retrieval miss = 3/20; gold in Top-5 but wrong = 9/20.

**FEVER 3B smoke test** (`results/base_rag/fever_smoke_test.jsonl`): 20 samples; classification accuracy = 0.8000; gold evidence in Top-5 = 20/20; retrieval miss = 0; gold in Top-5 but prediction wrong = 4. This run used a reduced controlled candidate corpus, so 20/20 retrieval hit is **not** a formal retrieval result.

## Quick Start

See [docs/SETUP.md](docs/SETUP.md).
