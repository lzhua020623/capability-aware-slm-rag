# NQ Train returned annotations

Place **fully labeled** delivery shards in this directory when annotation is
complete.

## Required files

Return the same 15 shard filenames as delivery:

```
nq_train_annotation_000.jsonl
...
nq_train_annotation_014.jsonl
```

Source delivery: `../nq_train_delivery/`

## Allowed edits

| Field | Rule |
|---|---|
| `label` | **Required.** One of: `correct`, `evidence_insufficiency`, `evidence_accessibility`, `evidence_utilisation` |
| `annotation_note` | Optional free-text; offline only; never used as Diagnosis runtime features |

## Forbidden edits

Do **not** change:

- `sample_id`
- `dataset`
- `query`
- `initial_answer`
- `evidence` (any Top-5 field)
- `annotation_reference` (any gold/reference field)

Any mutation of frozen fields causes validation to **FAIL** (no silent repair).

## Validate / merge

```powershell
.\.venv\Scripts\python.exe scripts/validate_nq_train_annotations.py
.\.venv\Scripts\python.exe scripts/validate_nq_train_annotations.py --merge
```

Successful merge writes:

- `../normalized/nq_train_74097_labels.jsonl`
- `../normalized/nq_train_74097_labels_summary.json`

Do not put unlabeled or partial shards here until the full 74,097 set is ready
unless your workflow explicitly validates incomplete returns (default validator
requires all 15 shards and all IDs).
