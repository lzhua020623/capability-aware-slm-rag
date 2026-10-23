# Preliminary Recoverability Experiment

Frozen configuration for the team experiment. Do not start from a new random sample, and do not change this protocol.

Formal config: `configs/preliminary.yaml`.  
Sample IDs: `configs/splits/preliminary_nq_500.json`, `configs/splits/preliminary_fever_500.json`.

FEVER500 7B C0 and C3 are complete; see [results and validation](../results/preliminary/fever_c0_c3_summary.md). FEVER C1 is still needed to compute Recoverable Failure Rate. This document continues to define the frozen protocol.

## Research question

Of ordinary Small-Model RAG failures, how many can the 7B model recover if it is given the correct evidence?

## Conditions

| Condition | Meaning |
|-----------|---------|
| **C0** | No evidence. Can the model answer from its own knowledge? |
| **C1** | Ordinary RAG: retrieved Top-5 evidence. |
| **C3** | Oracle: the correct gold evidence is given directly. |

**C2 is not part of this preliminary experiment.** C2 belongs to later product development and the final evaluation.

C0, C1, and C3 must use the **same** frozen sample IDs. Do not resample per condition.

## Frozen models and retrieval

- Primary model: `Qwen/Qwen2.5-7B-Instruct`, 4-bit NF4, `do_sample: false`, `max_new_tokens: 64`
- Shared retriever: `BAAI/bge-base-en-v1.5`, L2-normalized embeddings, inner-product similarity, Top-5
- **NQ C1** uses the existing full FAISS **IndexFlatIP**: `data/processed/indexes/nq_bge_base.index`
- **FEVER C1** uses the full `fever_corpus.jsonl` (5,396,106 passages) and a frozen FAISS **IndexIVFPQ**. It is **not** the earlier 100k controlled smoke-test pool; that pool is not valid for formal results
- Large FEVER corpus/index files are gitignored. Share them externally, or rebuild with `python scripts/build_fever_index.py`
- Retrieval settings in `configs/preliminary.yaml` must not be changed

## Larger-model control

- Required: `Qwen/Qwen2.5-32B-Instruct`
- Optional escalation: `Qwen/Qwen2.5-72B-Instruct`
- 32B runs **only** on samples that are C1 wrong **and** C3-7B wrong
- 72B, if resources allow, runs **only** on remaining C3-32B wrong samples
- 72B results must be reported separately and do not change the metric definitions below

## Frozen samples

- NQ: 500 unique IDs from the current 3216-sample controlled subset, `seed = 42`
- FEVER: 500 unique IDs from SUPPORTS/REFUTES samples with valid gold evidence, `seed = 42`, **250 SUPPORTS + 250 REFUTES**

## Frozen prompts

**NQ C0**

Answer the question using your knowledge. Return only the shortest answer span. Do not provide explanations or full sentences.

**NQ C1 / C3**

Answer the question using only the provided context. Return only the shortest answer span. Do not provide explanations or full sentences.

**FEVER C0**

Determine whether the claim is supported or refuted. Return only SUPPORTS or REFUTES.

**FEVER C1 / C3**

Based only on the provided evidence, determine whether the claim is supported or refuted. Return only SUPPORTS or REFUTES.

## Frozen evaluation

- NQ: normalized Exact Match and token-level F1, using the existing functions in `src/evaluation/qa.py` (same normalization as the verified NQ smoke tests)
- FEVER: classification accuracy
- Primary statistic: **Recoverable Failure Rate** = number(C1 wrong AND C3-7B correct) / number(C1 wrong)

Also record:

- C0 correct AND C1 wrong = retrieval-induced degradation
- C1 wrong AND C3-7B wrong AND C3-32B correct = likely small-model capability limitation
- 32B still wrong = unresolved / hard case

## Result records

Each condition, each sample, save at least: sample id, dataset, condition, model name, question or claim, gold answer / gold label, evidence (empty for C0), prediction, correct, latency.

NQ also save EM and F1. C1 also save retrieved Top-5 IDs and whether gold evidence is in Top-5.

Output paths are listed in `configs/preliminary.yaml`.

## Team split

- **Person A:** NQ 500, 7B C0 + C1
- **Person B:** FEVER 500, 7B C0 + C1
- **Person C:** the same NQ 500 + FEVER 500, 7B C3
- **Person D:** collect samples that are C1 wrong and C3-7B wrong from A/B/C, run C3 + 32B; if resources allow, run C3 + 72B only on samples that 32B still gets wrong

## Do not change

Nobody may independently change: sample IDs, model, retriever, Top-k, prompts, decoding, seed, or evaluation metrics.
