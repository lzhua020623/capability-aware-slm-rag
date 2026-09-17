# Frozen Base RAG retrieval config

This configuration is frozen from the NQ retrieval pilot. Formal Base RAG uses Top-5 retrieved passages.

## Frozen settings

From `configs/base.yaml`:

- `embedding_model`: `BAAI/bge-base-en-v1.5`
- `top_k`: `5`
- `index_type`: `faiss_flat_ip`
- `normalize_embeddings`: `true`

## NQ pilot results

Same evaluation protocol for both models: 3216 NQ queries, 99,922 passages, L2-normalized embeddings, FAISS inner-product (`IndexFlatIP`).

| Model | Recall@3 | Recall@5 | Recall@10 |
|-------|----------|----------|-----------|
| intfloat/e5-base-v2 | 0.6894 | 0.7705 | 0.8567 |
| BAAI/bge-base-en-v1.5 | 0.6872 | 0.7718 | 0.8451 |

The two models are close. Formal Base RAG uses Top-5, so `BAAI/bge-base-en-v1.5` is selected (BGE Recall@5 = 0.7718 vs E5 Recall@5 = 0.7705). Top-5 is a compromise among retrieval recall, context noise, and inference cost.

## Kept comparison artifacts

Do not delete the E5 index or metrics:

- `data/processed/indexes/nq_e5_base.index`
- `data/processed/indexes/nq_e5_base_meta.jsonl`
- `results/retrieval/nq_e5_metrics.json`

BGE artifacts used by Base RAG:

- `data/processed/indexes/nq_bge_base.index`
- `data/processed/indexes/nq_bge_base_meta.jsonl`
- `results/retrieval/nq_bge_metrics.json`
