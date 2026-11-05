# Intake of new NQ Train initial labels and E's usage boundaries

On 2026-10-22, 15 files from `NQ_C1_000_preliminary_annotations.xlsx` to `NQ_C1_014_preliminary_annotations.xlsx` were received from the [team Drive: NQ_15批](https://drive.google.com/drive/u/0/folders/1JWLByrI_hZ1qv70TSD23QjOlHngGBP1X), covering all of the original **74,097 NQ Train C1 items**. This resolves the problem that only the old empty-label delivery copy was available locally; it is not 74,097 new independent questions, nor is it FEVER data.

Complete results are in the [machine audit](../data/annotations/nq_train_preliminary/import_audit_20261022.json). The 15 workbooks, raw JSONL, labels, splits and human reviews are each stored by their own source, without overwriting old material; this round did not call the GPU, select new evaluation questions, or change prompts or strategy.

## Consistency and annotation sources

sample_id, dataset, query, initial_answer, evidence and annotation_reference were checked item by item. After deserialising nested JSON, all fields and types were compared; the SHA-256 of the raw JSONL matches the main delivery manifest. The 74,097 IDs are complete and unique, with no additions, omissions or field rewrites; all IDs in Excel are text, and no floating-point precision truncation occurred. Original evidence ranks, passage IDs, body text, scores and offline references all remain consistent.

The self-reported totals in the workbook Summary are:

| Source declaration | Count |
|---|---:|
| Automatic/heuristic initial labels | 73,536 |
| Selected-sample Top-5 semantic review | 561 |
| Total | 74,097 |

"Semantic review" is the files' source declaration and has not been verified as independent human review. The workbooks do not provide complete, verifiable per-item reviewers and statuses; some notes mention the first several rows, and some only say selected examples. Therefore this intake does not infer per-item human reviewer identity, and all labels are imported as **preliminary** with `human_verified=false`. This does not mean re-determining that none of them was looked at, but rather that there is not enough basis to upgrade them into a human gold standard.

Each Summary explicitly states that the automatic part uses answer/evidence lexical matching, approximate matching and retrieval signals, and that gold hits or lexical overlap cannot directly prove correctness, evidence sufficiency or error causation. Review_Questions has 37,703 rows in total, including processed records, and is used only to retain doubt tracking; it cannot be described as 37,703 still unresolved, nor can samples outside the sheet be regarded as confirmed correct.

## Initial-label distribution

The following are counts of file labels, **not verified error rates or C1 accuracy**.

| Initial label | development | calibration | reserved_evaluation | All |
|---|---:|---:|---:|---:|
| correct | 29,344 | 3,722 | 3,733 | 36,799 |
| evidence_insufficiency | 11,788 | 1,436 | 1,526 | 14,750 |
| evidence_accessibility | 14,410 | 1,765 | 1,842 | 18,017 |
| evidence_utilisation | 3,646 | 422 | 463 | 4,531 |
| Total | 59,188 | 7,345 | 7,564 | 74,097 |

The split established on 17 October is reused, without regrouping based on labels or results. Only per-purpose label counts are tallied here, and the content of reserved evaluation questions is not used to pick prompts. The existing 125-item challenge has been used and can no longer be called new unseen samples; team Diagnosis training exclusion and Wikipedia page and semantic near-duplicate isolation are still not confirmed.

## Cross-check against existing human review

Against the cumulative human review records of the first 100 items: of the 75 finalized, 39 have matching labels and 36 do not; the 25 originally deferred remain deferred. All existing human reviews are kept, and initial labels do not overwrite human results. With human-review labels as rows and initial labels as columns:

| Existing human review | correct | insufficiency | accessibility | utilisation | Total |
|---|---:|---:|---:|---:|---:|
| correct | 39 | 1 | 19 | 5 | 64 |
| evidence_utilisation | 1 | 0 | 10 | 0 | 11 |

In this reviewed development subset, of the 11 E errors, 10 were assigned accessibility and 1 correct by the initial labels, and none were assigned utilisation. Taking questions only by initial-label utilisation would miss these known errors; there are also 5 confirmed correct answers labelled utilisation by the initial labels.

This shows clear disagreement between the class boundaries of the initial labels and the current human review, especially since lexical matching cannot confirm the specific cause of evidence misuse. **36/75 cannot be extrapolated to a full-set label error rate**: the original 100 are a development review batch screened by EM/reference recall, not a random sample or independent blind review. Nor can it be declared from this that all 4,531 candidates are untrustworthy or all repairable. Next, stratified sampled review should first be done in development, including utilisation, accessibility, correct and insufficiency, to identify missed and wrong classifications; it cannot review only samples "judged as E".

## Impact on what comes next

1. This batch of labels is useful: it can expand the scope of retrieval, screening and stratified review, without re-labelling all 74,000 from scratch. Existing finalized samples are reused directly.
2. There is no need for now to replace NQ or do data augmentation. The current priority is label trustworthiness, the E output interface and evaluation isolation; new synthetic questions cannot replace real evaluation on the natural distribution.
3. The latest 10 questions are an engineering smoke: NQ 5 + FEVER 5, with three groups generating 30 predictions. They only verify execution, sources and output structure, cannot measure overall gain, and cannot be used repeatedly to pick prompts.
4. v9 Version 3's structural validity rate is 11/20, failing the pre-declared 18/20 threshold. This intake does not lift that stop condition; first set up a separate constrained-output interface engineering experiment, and only after it passes freeze larger development and unseen evaluations.
5. E's core validation needs trustworthy natural errors and correct controls, reporting fixes, harmful repairs, invalid fallbacks and cost. Stratified diagnostic sets and whole-set gains on the natural distribution should be reported separately, and a targeted error-picked set cannot be treated as overall accuracy.
6. All 74,097 are NQ. FEVER needs its data and diagnosis labels supplemented and verified separately; the FEVER sample count cannot be increased on this basis.

## Storage and reproduction

| Content | Path |
|---|---|
| Original ZIP, 15 XLSX files, per-file download manifest (local, Git-ignored) | `data/processed/external/nq_train_preliminary_20261022/` |
| Full standardised initial-label layer (local, Git-ignored) | `data/processed/answer_repair/nq_train_preliminary_20261022/labels.jsonl` |
| Committable audit, Summary original text, disagreement IDs and labels | `data/annotations/nq_train_preliminary/import_audit_20261022.json` |
| Read and verification script | `scripts/import_nq_train_preliminary.py` |
| Precision, source, frozen-field and overwrite-refusal checks | `tests/test_nq_train_preliminary.py` |

Original ZIP SHA-256: `fb7c0d3e404a24f306cc6f9e11763187e027cb2acff68c063bdfe102e3eb0742`. Standardised layer SHA-256: `1c08f65bd0351c66d27dd8ec6400083c5c2b898bcb95fa08c194aa8c3d9d4da1`. Each item exports only offline labels, purpose, source sheet/row number and hash, with `runtime_use=false`, excluding questions, evidence and gold; model input still uses the existing whitelist.

```sh
python scripts/import_nq_train_preliminary.py
python -m unittest tests.test_nq_train_preliminary -q
```

The script only reads XLSX and does not modify or re-export the original sheets. It first verifies all workbooks, source shards, frozen fields, IDs, labels, Summary and splits, then publishes results; replay with identical content is allowed, and overwriting existing different results is refused. The complete XLSX/ZIP/full JSONL are not included in Git and are reproduced from the audit fingerprints and the original Drive source.
