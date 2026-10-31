# FEVER baseline resources (contract only)

See `fever_resources_v1.json` for expected paths, SHA256, and present/missing status.

Large artifacts (corpus / FAISS index / ids / offsets) are **not** stored here.
Place matching files under the expected `data/processed/...` paths before FEVER RAG runs.
Qwen-only FEVER Test needs only the evaluation bundle:

- `baselines/data/fever_dev_13332.jsonl`
- `baselines/splits/fever_dev_13332_split.json`
