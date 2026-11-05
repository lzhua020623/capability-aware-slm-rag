# Data

The project uses Natural Questions and FEVER; FinDER is only an optional direction and currently has no corresponding formal pipeline.

| Directory | Contents | Git management |
|---|---|---|
| `raw/` | Official downloads, FEVER Wikipedia and raw NQ data | Data contents ignored, `.gitkeep` kept |
| `processed/` | Rebuildable artifacts such as controlled samples, retrieval corpus and indexes | Data contents ignored, `.gitkeep` kept |
| `annotations/assisted/` | Assisted review judgments and rationales from the assistant (agent) | Committed; must not be treated as human ground truth |
| `annotations/human/` | User confirmations, paired human fields and their sources | Committed; some records still pending review |
| `annotations/gold/` | Frozen reference answers for batch 4 of 300 NQ items | Committed; used only for offline evaluation |
| `annotations/nq_train_delivery/` | New NQ Train shard manifest, source notes and Drive check report | Small metadata committed; the original 15 shards are in local `processed/external/` |

On 2026-10-17 all 74,097 NQ Train C1 items were integrated, with all labels null. The full E-compatible input is in local `processed/answer_repair/nq_train_v1/c1_scored.jsonl`, ignored by Git; the first batch of 100 development review materials and the internal split configuration are committed; see [Integration notes](../docs/ANSWER_REPAIR_NQ_TRAIN.md). 100 is a review batch, not the dataset total, and screening groups are not the same as human diagnosis labels.

Local `.xlsx` workbooks are ignored, but may contain upstream annotations and gold sources that have not yet been fully exported, so trustworthy backups should be kept. Raw data not being uploaded to GitHub does not mean the repository has no annotations or reference answers.

The normalized diagnosis labels for the 500 project samples each of NQ and FEVER and for the NQ3216 extension pool are in the Diagnosis branch; see [Project status](../docs/PROJECT_STATUS_AND_DIRECTION.md). Upstream AI initial labels must be distinguished from human confirmation. For data acquisition and processing see [Setup instructions](../docs/SETUP.md); the E runtime uses only the query, the C1 initial answer and the retrieval evidence, and does not read gold.
