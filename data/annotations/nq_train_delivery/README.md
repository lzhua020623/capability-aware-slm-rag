# NQ Train Diagnosis annotation delivery

This package is ready for offline Diagnosis primary-label annotation.

## Contents

- **15** JSONL shards: `nq_train_annotation_000.jsonl` … `nq_train_annotation_014.jsonl`
- **14 × 5000 + 1 × 4097 = 74,097** samples
- `manifest.json` (per-shard counts, sample_id ranges, SHA256)
- Annotation semantics: `docs/NQ_TRAIN_DIAGNOSIS_ANNOTATION_GUIDE.md` (repo root)

## What to edit

Fill **only** the `label` field on every row.

Optional (offline only, not a Diagnosis runtime feature): `annotation_note`.

### Allowed `label` values

- `correct`
- `evidence_insufficiency`
- `evidence_accessibility`
- `evidence_utilisation`

## Do not modify

These fields are frozen:

- `sample_id`
- `dataset`
- `query`
- `initial_answer`
- `evidence`
- `annotation_reference`

Changing any of them will cause validation to fail.

## How to return

1. Keep the same 15 shard filenames.
2. Put the completed shards into:

   `data/annotations/nq_train_returned/`

3. The project owner will run:

   ```text
   python scripts/validate_nq_train_annotations.py
   python scripts/validate_nq_train_annotations.py --merge
   ```
