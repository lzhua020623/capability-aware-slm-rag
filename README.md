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
- Preliminary Recoverability Experiment configuration is frozen; FEVER500 7B C0 and C3 are complete: **72.0%** and **95.2%** accuracy. See [results and validation](results/preliminary/fever_c0_c3_summary.md). FEVER C1 is still needed for Recoverable Failure Rate.
- see [docs/PRELIMINARY_EXPERIMENT.md](docs/PRELIMINARY_EXPERIMENT.md)

## Engineering Validation

These are **engineering smoke tests** only. They do **not** answer the Preliminary Recoverability Experiment research questions.

**NQ 3B historical smoke test v2** (`results/base_rag/nq_smoke_test_v2.jsonl`): 20 samples; EM = 0.3000; Average F1 = 0.3460; gold evidence in Top-5 = 17/20; retrieval miss = 3/20; gold in Top-5 but wrong = 11/20.

**NQ 7B smoke test** (`results/base_rag/nq_smoke_test_7b.jsonl`): 20 samples; EM = 0.4000; Average F1 = 0.4770; gold evidence in Top-5 = 17/20; retrieval miss = 3/20; gold in Top-5 but wrong = 9/20.

**FEVER 3B smoke test** (`results/base_rag/fever_smoke_test.jsonl`): 20 samples; classification accuracy = 0.8000; gold evidence in Top-5 = 20/20; retrieval miss = 0; gold in Top-5 but prediction wrong = 4. This run used a reduced controlled candidate corpus, so 20/20 retrieval hit is **not** a formal retrieval result.

## Quick Start

See [docs/SETUP.md](docs/SETUP.md).

FEVER500 7B C0 / C3: [运行与续跑说明](docs/FEVER_C0_C3.md).
