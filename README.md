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
- Preliminary Recoverability Experiment 7B runs completed: NQ C0/C1/C3 and FEVER C1/C3, 500 frozen samples each
- larger-model control completed on this branch (`feature/-ce-70B`)
- see [docs/PRELIMINARY_EXPERIMENT.md](docs/PRELIMINARY_EXPERIMENT.md) and [docs/LARGER_MODEL_CONTROL.md](docs/LARGER_MODEL_CONTROL.md)

## Engineering Validation

These are **engineering smoke tests** only. They do **not** answer the Preliminary Recoverability Experiment research questions.

**NQ 3B historical smoke test v2** (`results/base_rag/nq_smoke_test_v2.jsonl`): 20 samples; EM = 0.3000; Average F1 = 0.3460; gold evidence in Top-5 = 17/20; retrieval miss = 3/20; gold in Top-5 but wrong = 11/20.

**NQ 7B smoke test** (`results/base_rag/nq_smoke_test_7b.jsonl`): 20 samples; EM = 0.4000; Average F1 = 0.4770; gold evidence in Top-5 = 17/20; retrieval miss = 3/20; gold in Top-5 but wrong = 9/20.

**FEVER 3B smoke test** (`results/base_rag/fever_smoke_test.jsonl`): 20 samples; classification accuracy = 0.8000; gold evidence in Top-5 = 20/20; retrieval miss = 0; gold in Top-5 but prediction wrong = 4. This run used a reduced controlled candidate corpus, so 20/20 retrieval hit is **not** a formal retrieval result.

## Preliminary Recoverability Experiment: 7B results

500 frozen samples per dataset, `Qwen/Qwen2.5-7B-Instruct`, 4-bit NF4.

| | C0 no evidence | C1 retrieved Top-5 | C3 gold evidence |
|---|---|---|---|
| NQ (EM) | 0.102 | 0.352 | 0.516 |
| NQ (token F1) | 0.194 | 0.490 | 0.678 |
| FEVER (accuracy) | — | 0.804 | 0.952 |

Retrieval helps: 137 NQ samples go from C0 wrong to C1 correct, against 12 that
degrade the other way.

**Recoverable Failure Rate** — C1 wrong and C3 correct, over C1 wrong:

| | NQ | FEVER |
|---|---|---|
| C1 wrong | 324 | 98 |
| recovered by gold evidence | 97 | 80 |
| **RFR** | **29.9%** | **81.6%** |
| persistent failures (C1 and C3 both wrong) | 227 | 18 |

## Larger-model control: Llama-3.3-70B

`meta-llama/llama-3.3-70b-instruct` under C3, on the samples the 7B failed under
both C1 and C3. Full write-up in [docs/LARGER_MODEL_CONTROL.md](docs/LARGER_MODEL_CONTROL.md).

| Dataset | Cohort | Recovered | 95% CI |
|---|---|---|---|
| NQ | 67 | **15 (22.4%)** | [14.1%, 33.7%] |
| FEVER | 18 | **8 (44.4%)** | [24.6%, 66.3%] |
| Combined | 85 | **23 (27.1%)** | [18.8%, 37.3%] |

23 samples are a likely small-model capability limitation; **62 are unresolved
hard cases** where a 70B with the correct evidence still fails, so model scale is
not what is missing.

Every sample was selected because the 7B got it wrong, so these support "of the
7B's failures the 70B recovers 27%" and **no** claim about which model is more
accurate. Measuring that needs a control group of 7B-correct samples.

### Exact match understates both models on NQ

The frozen NQ metric is normalized exact match against a single gold string, but
the stored golds are often whole clauses (median 5 words, p75 11, max 41) while
the frozen prompt asks for "only the shortest answer span" — the model obeys the
instruction and fails the metric. A `Qwen2.5-72B` judge over all 566
exact-match-wrong answers accepted exactly half, and agreed with exact match on
60/60 samples it had already scored correct.

| | exact match | LLM judge |
|---|---|---|
| NQ C1 accuracy | 35.2% | 61.6% |
| NQ C3 accuracy | 51.6% | 81.8% |
| **NQ RFR** | **29.9%** [25.2, 35.1] | **65.1%** [58.1, 71.5] |
| NQ persistent failures | 227 | **67** |

The intervals do not overlap. This is a **re-analysis, not a protocol change**:
the frozen exact-match numbers stay the headline unless the team agrees to change
the metric. Both the 7B and the 70B were scored by the same judge, since scoring
two models by different rules makes the comparison meaningless. FEVER needs no
correction — its predictions parse cleanly to SUPPORTS/REFUTES.

Judge-corrected cohorts are written alongside the frozen ones with a `_judged`
suffix; the frozen files are never overwritten.

## Quick Start

See [docs/SETUP.md](docs/SETUP.md). To verify the environment without downloading
any dataset, run `python scripts/verify_env.py`.
