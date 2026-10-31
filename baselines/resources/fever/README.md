# FEVER baseline resources (contract only)

See `fever_resources_v1.json` for expected paths, SHA256, and present/missing status.

## Evaluation bundle (committed)

- `baselines/data/fever_dev_13332.jsonl`
- `baselines/splits/fever_dev_13332_split.json`

Qwen-only needs only this bundle. FEVER RAG also needs local retrieval assets below.

## Large artifacts (do **not** commit)

Place under the expected paths before FEVER RAG:

| logical_name | expected path |
|---|---|
| fever_corpus | `data/processed/retrieval/fever_corpus.jsonl` |
| fever_bge_ivfpq_index | `data/processed/indexes/fever_bge_ivfpq.index` |
| fever_bge_ivfpq_ids | `data/processed/indexes/fever_bge_ivfpq_ids.jsonl` |
| fever_bge_ivfpq_offsets | `data/processed/indexes/fever_bge_ivfpq_offsets.npy` |

Retriever: `BAAI/bge-base-en-v1.5`, IndexIVFPQ, `nprobe=16`, Top-k=5.

### Offsets rebuild (deterministic; index untouched)

`fever_bge_ivfpq_offsets.npy` is a byte-offset sidecar into `fever_corpus.jsonl` in
**ids JSONL order**. It can be rebuilt locally without retraining:

1. Load passage_ids from `fever_bge_ivfpq_ids.jsonl` (n=5396106).
2. Scan `fever_corpus.jsonl` sequentially; record `tell()` before each line.
3. Assert `corpus[i].passage_id == ids[i]` (spot-check every 100k).
4. `np.save(..., dtype=int64)`.

Do **not** rebuild the FAISS index for baseline reuse.

### Metadata

Canonical small meta file: `results/retrieval/fever_bge_ivfpq_meta.json`
(tracked when present). Describes index hyperparameters and expected `ntotal`.
