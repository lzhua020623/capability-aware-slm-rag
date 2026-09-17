# Frozen Reference RAG retrieval config

These numbers are **retrieval pilot** results on the NQ controlled subset. They are used to freeze the Reference RAG retriever. They are **not** the final project evaluation.

## Frozen settings

From `configs/base.yaml`:

- `embedding_model`: `BAAI/bge-base-en-v1.5`
- `top_k`: `5`
- `index_type`: `faiss_flat_ip`
- `normalize_embeddings`: `true`

Formal Reference RAG uses **BGE + FAISS + Top-5**.

## NQ retrieval pilot

Same protocol for both models: 3216 NQ queries, 99,922 passages, L2-normalized embeddings, FAISS inner-product (`IndexFlatIP`).

| Model | Recall@3 | Recall@5 | Recall@10 |
|-------|----------|----------|-----------|
| intfloat/e5-base-v2 | 0.6894 | 0.7705 | 0.8567 |
| BAAI/bge-base-en-v1.5 | 0.6872 | 0.7718 | 0.8451 |

## Conclusion

The two models are very close overall. Formal Reference RAG uses Top-5. BGE is slightly higher on Recall@5 (0.7718 vs 0.7705), so the frozen retriever is `BAAI/bge-base-en-v1.5` + FAISS + Top-5. This does **not** claim that BGE is substantially better than E5.

## Kept comparison artifacts

Do not delete or overwrite these files:

- `data/processed/indexes/nq_e5_base.index`
- `data/processed/indexes/nq_e5_base_meta.jsonl`
- `results/retrieval/nq_e5_metrics.json`

BGE artifacts used by Reference RAG:

- `data/processed/indexes/nq_bge_base.index`
- `data/processed/indexes/nq_bge_base_meta.jsonl`
- `results/retrieval/nq_bge_metrics.json`
