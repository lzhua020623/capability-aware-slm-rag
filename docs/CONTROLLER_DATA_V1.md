# Controller data freeze v1 · T2

Date: 2026-10-23. Corresponds to the [T1 experiment protocol](CONTROLLER_EXPERIMENT_PROTOCOL_V1.md).

2026-10-24 C1 completion: the new FEVER 900 items have been collected uniformly and passed input/output and environment acceptance; for the runtime and fingerprints see [fever_c1_completed_v1.json](../configs/controller/fever_c1_completed_v1.json), and for the completion report see [completed_v3](../artifacts/controller/fever_c1_v1/prefill_r2/completed_v3/README.md). The text below and the original T2 manifest keep their state at freeze time, without overwriting their hashes; this new registration only proves that FEVER C1 is usable, and the formal Controller comparison must still pass T3 execution acceptance. The existing NQ C1, the original split and the complete held-out pool are unchanged.

2026-10-24 resource progress: the NQ Train original corpus and original index have been retrieved from the team Drive, and [the IDs/offsets were restored together with a restoration record](NQ_TRAIN_INDEX_RECOVERY.md); 64 independent vector comparisons passed. This document and `data_v1/asset_inventory.json` keep the historical state at the T2 freeze; the current NQ resources are as given by the new restoration archive. The FEVER resource package contents, the official execution environment and the T3 provider adaptation are still pending acceptance, and T2 is not marked as fully ready on this basis.

**Samples, the complete evaluation held-out pool, paired references and exposure records are frozen. T2 has not yet been fully accepted: the new FEVER C1 and matching retrieval resources are still to be completed.** This round involved no inference, training, answer changes or computation of new model accuracy.

## 1. Both datasets hold out 10%–15% for evaluation

| Dataset | Source population this round | Complete evaluation held-out pool | Proportion | Fixed evaluation subset this round | Remainder still sealed |
|---|---:|---:|---:|---:|---:|
| NQ Train | 74,097 | **7,564** | **10.21%** | 500 | 7,064 |
| FEVER | 13,229 binary-classification candidates with valid gold evidence | **1,886** | **14.26%** | 500 | 1,386 |

**500 is the pre-registered actual evaluation subset for this round, not the entire held-out amount. All held-out IDs are forbidden for training, prompt modification, threshold tuning and candidate selection.** The remaining held-out samples that were not drawn cannot be put back into the training pool; nor can the sample be expanded ad hoc after seeing the results of the 500 until the desired significance is obtained. To switch to evaluation on the complete held-out pool, a new run scope and budget must be registered before the first output.

The complete held-out IDs are in [evaluation_reserve_ids.json](../artifacts/controller/data_v1/evaluation_reserve_ids.json), and the team's unified entry point is [controller_data_v1.json](../configs/splits/controller_data_v1.json). The held-out pool is a usage restriction; it does not mean the whole pool has undergone human semantic review, page isolation or C1 resource acceptance. Held-out samples isolated by quality rules still must not be moved into training.

NQ keeps its original internal usage: development 59,188, calibration 7,345, reserved_evaluation 7,564; the historical split is unchanged. The Controller's `selection` corresponds to the original calibration. FEVER assigns 70/15/15 usages over valid candidates by question hash, then excludes already-exposed IDs, yielding 8,956 development candidates, 1,869 selection-run candidates and a complete held-out pool of 1,886. FEVER's proportion uses the 13,229 valid candidates as the denominator, **not the size of the complete FEVER train**.

## 2. Run list for this round

| Usage | NQ | FEVER | May be used for optimisation |
|---|---:|---:|---|
| development | 200 | 200 | May be used to locate errors and design the single candidate |
| selection | 200 | 200 | Only for the selection-run comparison between the frozen candidate and the baseline; no further per-question prompt changes |
| final_candidate | 500 | 500 | Final-version evaluation only; this directory name indicates that C1/resource sealing is still required before execution |
| Total | 900 | 900 | Not full-scale training or full-scale evaluation |

All IDs, order and SHAs are in [manifest.json](../artifacts/controller/data_v1/manifest.json) and [selection_ids.json](../artifacts/controller/data_v1/selection_ids.json). 900 is the total across the three usages and cannot be combined and reported as 900 unseen test scores. Nor can the 74,097 source-data size be written up as the number of experimental samples.

500 is a pre-registered choice to control the inference scale of this round and cannot guarantee detecting a small gain of 1–2 percentage points. As a planning example, the 95% normal-approximation half-width for a single accuracy under the least favourable distribution is about 4.38 percentage points; the interval for a paired difference depends on the two models' disagreement rate, and assuming a 20% disagreement rate it is about 3.92 percentage points. These are not experimental results, nor a power guarantee. The T1 paired intervals and tests are final.

## 3. Filtering and exposure isolation

- The 15 original NQ shards each passed the delivery SHA and count check, 74,097 items in total; the originally delivered short-answer references are used, and initial (AI) labels are not read as sampling scores.
- FEVER excludes 103 items from the 13,332 local binary-classification records under the existing "valid gold evidence" condition, leaving 13,229; questions are not selected by model correctness or expected repairability.
- Checked 525 historical JSON/JSONL files in reference commit `30738f0`, covering the original C1/enhancement results, review records, Diagnosis splits and existing exposure registrations. The NQ Train pool hit 426 exposed IDs; the FEVER valid pool hit 518. All were excluded from this round's three sample groups.
- NQ isolates 35 question variants from the old audit, and isolates a further 46 cross-usage variants by NFKC, casefold, punctuation removal and whitespace normalisation. The original split files and reports are retained.
- Drawn by SHA ordering with seed 42, protecting the final subset first, then taking selection run and development. A direct lexical near-match check of content-token Jaccard ≥0.60 is also done between candidates and exposed questions and already-selected questions; no semantic inference is made, and gold/C1 correctness is not read for ordering or substitution.
- The user explicitly confirmed in this round that the team has not used this data for training, tuning or error analysis outside the repository records; this statement is saved in the policy/exposure audit. "Unexposed" here is limited to the project records and this statement, and does not claim to exclude model pre-training, semantic near-matches or identical Wikipedia pages.

Full records are in [exposure_audit.json](../artifacts/controller/data_v1/exposure_audit.json) and [exclusions.jsonl](../artifacts/controller/data_v1/exclusions.jsonl). Lexical filtering changed the candidate distribution; reports should state that this is an evaluation subset controlled by this project, not an unconditional score on the whole official dataset.

## 4. Existing C1 and what is still missing

NQ exports 900 strict runtime records containing only `sample_id,dataset,query,initial_answer,evidence`, with gold saved separately. Each question keeps the original C1 and the ordered Top5. The original C1's Qwen revision/full environment was not exported, so this round explicitly adopts a "historical C1 post-processing" comparison, with the field kept as unknown; it will not be fabricated as a verified new revision.

FEVER exports 900 **C1 generation tasks containing only questions and IDs**, with their gold also saved separately. No `initial_answer` or Top5 was forged for them. New C1 must be generated with the T1-fixed Qwen and a matching FEVER index, and sealed, before the six-group enhancement comparison begins. The existing 500 old C1 items cannot substitute for this batch of new selection-run/final questions.

| Path | Contents |
|---|---|
| `artifacts/controller/data_v1/nq/<role>/runtime.jsonl` | NQ pure runtime inputs, usable by the Controller |
| `artifacts/controller/data_v1/fever/<role>/c1_tasks.jsonl` | FEVER tasks awaiting C1 generation; cannot yet serve as DiagnosisInput |
| `artifacts/controller/data_v1/offline/<dataset>/<role>/gold.jsonl` | Original-data reference answers/labels, only for the independent evaluation process |
| `evaluation_reserve_ids.json` | Complete held-out pools for both datasets; future training preparation must exclude all IDs |
| `policy.json` / `manifest.json` | Sampling, usage, hashes, provenance and the conditions still not completed |

Selection went through automatic structural checks only; no new human gold was added, and final questions were not opened for per-question discussion. The human-type-triggered control still needs reliable type annotation; new AI initial labels cannot automatically be treated as ground truth for that control. This does not prevent completing the integration engineering first.

## 5. All remote branches have been re-searched

This round checked the file trees of the 10 original remote branches plus the new `controller`, 11 branches in total. The record is in [branch_resource_inventory.json](../artifacts/controller/data_v1_validation/branch_resource_inventory.json).

| Found in branches | Path / notes |
|---|---|
| NQ Train C1 generation and export code | `scripts/run_nq_train_c1_generation.py`, on main and other branches, merged into controller |
| NQ Train C1 full statistics and delivery manifest | `data/annotations/nq_train_annotation_candidates_stats.json`, `data/annotations/nq_train_delivery/manifest.json`; records 74,097 items |
| FEVER C1 runner | `scripts/run_fever_c1_preliminary.py`, from feature/fever-c1 |
| FEVER old 500 predictions and Top5 | `results/preliminary/fever_c1_7b.jsonl`, `fever_c1_top5.jsonl` |
| FEVER index description file | `results/retrieval/fever_bge_ivfpq_meta.json`; describes 5,396,106 items, IVFPQ; the file itself is not the index |
| NQ Train index build and read code | `scripts/build_nq_train_bge_index.py`, `scripts/run_nq_train_bge_retriever.py` |

No branch tracks `.index`/`.npy` files; [SETUP](SETUP.md) and the [preliminary experiment protocol](PRELIMINARY_EXPERIMENT.md) state clearly that large corpora/indexes are not in Git and must be obtained from the original holders or rebuilt per the protocol. The complete NQ delivery already exists on this machine and passed manifest verification; it cannot be called "no C1 data" just because it is ignored by Git.

The following resources from the original runtime environment are still needed:

| NQ Train (1,224,119 passages) | FEVER (5,396,106 pages) |
|---|---|
| `nq_train_bge_base.index` | `fever_bge_ivfpq.index` |
| `nq_train_bge_base_ids.jsonl` | `fever_bge_ivfpq_ids.jsonl` |
| `nq_train_bge_base_offsets.npy` | `fever_corpus.jsonl` |
| `nq_train_corpus.jsonl` | Metadata exists, but must be checked against the actual files |
| `nq_train_bge_metadata.json` | BGE weight snapshot/revision matching C1 |
| BGE weight snapshot/revision matching C1 | |

The local FEVER corpus is a cloud placeholder file of about 3.52 GB; the placeholder state is not counted as full text being available. NQ Train resources use an ID/offset/corpus combination, while the current Controller provider interface expects metadata JSONL with full text; T3 must add lossless adaptation, and cannot pass off the ID file as full-text metadata, nor switch to the old NQ dev 99,922-passage index.

This round's Google Drive search returned `ACCESS_TOKEN_SCOPE_INSUFFICIENT`, and the browser fallback returned `No browser is available`; this is why remote resource discovery is not yet complete, and it cannot be used to assert that classmates did not save these files. Until the resources are in place, index matching, BGE revision and full-corpus quality remain unverified.

## 6. Verification and next steps

```bash
python scripts/check_controller_data.py --check-sources
python scripts/check_controller_data.py --require-full-t2
```

The first check covers the registered manifest, all artifact SHAs, source SHAs, paired IDs, strict field isolation, held-out proportions, and that held-out IDs do not enter optimisation sets. The second currently returns 2 as expected, meaning the new FEVER C1/resource seal is not complete. Data structure checks do not equal quality improvement or index acceptance.

Full CPU regression **665 passed** (17 added this round), see the [verification directory](../artifacts/controller/data_v1_validation/README.md). Original human review files, historical data, the T1 protocol and algorithm prompts are unchanged.

Next, T3 engineering for Full, No Probe, the human-type interface, data resource adaptation and evaluation acceptance can continue; before actually starting GPU paired comparison, matching resources must be obtained, FEVER C1 generated and sealed, and the independent execution checklist passed. This round does not require redoing human annotation first.
