# Team data alignment and usage boundaries · 2026-10-27

Current branch `controller_2`. This round ingests the team's original split as original bytes and tightens usage checks for later data preparation, training and the current production entry. There was no re-splitting, no modification of test IDs, and no training or teacher API was started.

## The team's single NQ split

Source: `feature/baseline-evaluation@387ddafc82bc939ebd27b3137d580db6ecb2e04a`.

- File: [diagnosis_nq_train_74097_split.json](../baselines/splits/diagnosis_nq_train_74097_split.json)
- SHA-256: `7da01c94cd7b5f92076edebc80911ac891d3861a01cf6b40a1e5e6efb6482dc4`
- Registration: [team_nq_data_v1.json](../configs/splits/team_nq_data_v1.json), whose hash is bound by the current production configuration.

| Usage | Team original count | New experiment pool conservatively allowed this round |
|---|---:|---:|
| Train | 51,866 | 46,421 |
| Validation | 11,117 | 9,440 |
| Test | 11,114 | Original IDs, order and count are all kept; formal execution requires a separate T7 seal |

The extra exclusions are usage restrictions, not re-splitting: Train still excludes the old full held-out pool and the old Controller selection; Validation additionally excludes the registered downstream development/analysis exposure. Excluded questions are not reassigned to another role.

## What was actually checked

- The local annotation index and the team split both have 74,097 unique IDs, with 0 missing, extra or duplicate; the team's three ID groups are disjoint.
- The label counts of the local initial labels aggregated by team Train/Validation exactly match the corresponding metadata in the split file; this cannot prove that each semantic label is correct, nor replace a per-ID label comparison against the original Diagnosis files.
- All 15 raw C1 shards are in `data/processed/external/nq_train_delivery_20261017/`, and each SHA-256 matches the committed delivery manifest. The raw dataset has nothing missing and does not need to be re-downloaded.
- Streaming checks on the allowed 46,421 Train and 9,440 Validation items: query, original answer, Top-5 count/order/unique IDs/original text/finite scores, and gold reference fields have no structural anomalies; duplicates of normalised query fingerprints across these two groups are 0.
- The questions, evidence and gold of Test are not decoded for checking. Near-duplicate screening was not redone in a way that reads test content; the team's original grouping declaration is not treated as a new independent content audit.

Independent re-check tool:

```bash
python scripts/audit_team_nq_data.py --output-dir /tmp/team-nq-audit-new
python scripts/audit_team_nq_data.py --ids-only --output-dir /tmp/team-nq-ids-new
```

The output directory must be new. The full check decodes only the C1 content of allowed roles; test/protected records are skipped by ID first. The tool does not modify data or models.

## Historical exposure expansion check

[known_exposure_ids.json](../artifacts/controller/team_data_v1/known_exposure_ids.json) is recomputed from the saved E, Controller and reranking experiment IDs, recording file hashes and provenance.

- The 130 items from the old E registry and the Controller development/selection run were only the scope of the previous check.
- After adding saved results such as reranking target classes, correct classes and pilot, a total of **603 distinct team test IDs** have entered downstream experiments or analysis. They are not "603 answered wrongly", nor do they equal "603 trained on".
- The remaining **10,511 items** only mean "not in the currently registered exposure sources", not proof that they are completely unseen by the team. The actual training/tuning IDs of the old training weights have not all been delivered, and potential training overlap has still not been ruled out.
- The final main table continues to align with the team's 11,114 items, discloses known exposure, and additionally reports a paired analysis excluding known exposure; it is not quietly replaced with another test set.

## Executable boundaries

Implementation: [src/data/team_nq.py](../src/data/team_nq.py).

1. Pin the original Git commit and split fingerprint; even if a reassignment keeps the same counts and the configuration hash is then recomputed, impersonating the original team split is rejected.
2. IDs must be exact strings; unknown IDs, float/integer coercion, cross-role, the old held-out pool and exposed Validation are all rejected. Independent external data needs separate provenance registration.
3. Split by team role first, then generate training positive/negative examples; borrowed answers for the answer selector come only from the same role, recording donor_sample_id. Synthetic/augmented samples need a verifiable parent ID.
4. The two builders no longer do another random 70/15/15 or 90/10 split on the 74,097 items. New data outputs contain only train/validation, with no test.jsonl, and do not overwrite old directories.
5. Before any model loading, CUDA setup or sample truncation, the two trainers verify the policy source file fingerprint, the full training file hash, roles, donor/parent and normalised question intersections. The builders additionally record the raw input file fingerprints. Old data without receipts must be re-prepared.
6. The reranking trainer does not read or automatically score the local test; reranking comparison and end-to-end component evaluation default to the registered Validation. Custom subsets, reused original-answer caches and old parameters cannot bypass usage checks; caches also check IDs first, before decoding original answers and loading models.
7. The current production entry defaults to data_role=validation, checks the role per question before Diagnosis/enhancement and records the policy/split fingerprints. The ordinary entry cannot request test; FEVER also cannot reuse NQ's roles before it has a unified registration. Low-level fabricated test injection is explicitly marked as not enforcing the production data boundary.
8. Future teacher inputs keep only sample_id, dataset, query, initial_answer, evidence via teacher_input(); gold, AI labels and extra evidence annotations do not enter requests. Teacher data generation is limited to the allowed Train.

Old frozen packages can still be reproduced under their respective historical protocols, without rewriting historical configurations and results; the above are the new experiment entries of the current branch. If P60 actually used an independent NQ Dev for training, it needs separate provenance registration and cross-set isolation, and cannot be disguised as this round's Train data.

## Data health check does not equal semantic acceptance

These Diagnosis labels are still initial AI labels. Semantic equivalence between original answers and gold, qualifiers, weakly supervised negatives, and whether the answer selector's gold positives are supported by evidence have not become human gold just because structural checks passed. Newly generated data receipts keep labels_semantically_verified=false; they cannot be used directly as high-quality E distillation targets.

The historical 74,097-item C1 and the new team baseline's context_fallback outputs are also not the same set of predictions. The former can be used for registered development/training; the predictions, metrics and run_config used for the final paired comparison are already in `baselines/outputs/`: Qwen-only, the companion RAG and the adaptive system with 11,114 items each, with Judge in `baselines/outputs/judge/`.

The retrieval corpus and index remain matched with the baseline. Usage isolation targets QA samples and supervision fields; normal retrieval passages related to test questions are not deleted from the Wikipedia knowledge base.

Team splitting for FEVER, P60's raw online features and model usage IDs, and the training provenance of candidate reranking/selectors are still later to-dos. The 11,114 per-question results and Judge of the new baseline are already in `baselines/outputs/`. generation_repair in the formal test has been executed; this round's health check did not enable new training weights.
