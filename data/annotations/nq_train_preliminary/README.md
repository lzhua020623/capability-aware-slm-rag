# NQ Train initial label integration

On 2026-10-22, 15 files `NQ_C1_000` to `014_preliminary_annotations.xlsx` were received from the team Drive folder [NQ_15批](https://drive.google.com/drive/u/0/folders/1JWLByrI_hZ1qv70TSD23QjOlHngGBP1X). This batch is a **new initial-label layer** on the original 74,097 items and adds no independent samples.

- [Per-file verification and source notes](import_audit_20261022.json): SHA-256, full label counts, the existing split distribution, the original workbook Summary text, and disagreements with the existing human review.
- Local raw files: `data/processed/external/nq_train_preliminary_20261022/`, containing the ZIP, the 15 unmodified XLSX files and the download manifest.
- Local full standardized layer: `data/processed/answer_repair/nq_train_preliminary_20261022/labels.jsonl`. Linked by ID, keeping the original labels, file hashes, sheet and row numbers; contains no questions, evidence or gold, `runtime_use=false`.
- Label mapping: `Correct`→`correct`; `Evidence Insufficiency`→`evidence_insufficiency`; `Evidence Accessibility Failure`→`evidence_accessibility`; `Evidence Utilisation Failure`→`evidence_utilisation`.

The workbooks state that 73,536 items are automatic heuristic initial labels and 561 underwent semantic review of selected samples; no per-item reviewer or review status sufficient for verification was provided. Therefore all are stored as preliminary, and no new `human_verified=true` labels were imported. These labels cannot be treated as human gold, model accuracy or a definitive error distribution.

This directory is separate from the human annotation return workflow of `nq_train_returned/`. The original JSONL, the human finalizations, the 25 deferred items and the experiment frozen packages were not modified. Of the 75 existing human reviews, 36 disagree with the initial labels, and the existing human review still prevails; already-handled human review items are not reopened. For details see the [Integration report](../../../docs/ANSWER_REPAIR_NQ_PRELIMINARY_20261022.md).

Reproduction: first download the above workbooks from Drive, restore the local download manifest and the 15 raw JSONL files, then run `python scripts/import_nq_train_preliminary.py`. Any ID, frozen field, hash, count or label anomaly causes publication to be refused. Identical output can be rerun; different content is refused from overwriting.
