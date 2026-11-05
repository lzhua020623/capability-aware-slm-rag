# Controller resource integration · 2026-10-24

Follow-up progress: the FEVER provider, fixed-query compatibility replay and the new 900-item C1 collection package have been completed; see [FEVER integration and collection](CONTROLLER_FEVER_C1_V1.md). The scope and acceptance numbers of this resource integration batch are kept below.

What is currently completed is the T2/T3 resource and NQ Train interface integration. It solves "how to fetch the corresponding original text after a vector hit", and does not change Diagnosis, Probe, routing thresholds, enhancement prompts or scoring criteria. There is no new QA quality improvement conclusion; the full comparison still requires the execution seal and GPU acceptance.

## Completed checks

| Object | Completed | Scope limits |
|---|---|---|
| NQ Train retrieval store | 1,224,119 IDs, offsets and original texts checked one by one, each with an actual seek read-back; bound to the original index SHA-256 | This round does not re-encode all vectors; the earlier independent BGE comparison of 64 passages is still recorded separately |
| NQ Controller provider | Explicit resource registry, file fingerprints and full check report accepted; original text read by offset, cache of at most 128 passages | Real Qwen routing and answer generation still require CUDA; CPU replay only accepts retrieval |
| FEVER original resources | Two original ZIPs downloaded from the team Drive; 5,396,106 corpus/ID entries fully checked in order, offsets generated | Original BGE revision not recorded; structural checks cannot be called semantic vector validation |
| FEVER original index | Metadata byte-identical to the record already in the repository; actual FAISS attribute and original-text search read-back checks archived separately | FEVER dynamic retrieval enhancement not yet integrated, and the new 900-item C1 has not been generated either |

Both counts are **knowledge-base passage counts**. NQ questions still come from the original 74,097-item subset. The frozen development/selection-run/final candidates are still 200/200/500 per dataset; the full held-out pools of NQ 7,564 and FEVER 1,886 are unchanged.

The independent vector validation snapshot for NQ is `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`. This is the snapshot explicitly pinned for this validation and new executions; the revision of the historical C1 is still unknown. The retrieval replay uses the first 10 questions of the already-exposed development set, comparing the Top5 IDs produced by real BGE queries against the original index, and also checks that the first five items of Top20 match; it does not read the offline gold, does not generate answers, and does not select samples or modify the algorithm by EM.

In the actual replay, **the Top5 IDs and their order match the original C1 for 10/10 questions**, and the first five items of Top20 also match. See the report in [retrieval_replay.json](../artifacts/controller/nq_mapping_full_v1/retrieval_replay.json). This is a retrieval resource/interface consistency result and cannot be converted into answer accuracy or a gain from E.

## NQ Train integration method

```bash
python scripts/run_controller_integration.py \
  --input artifacts/controller/data_v1/nq/development/runtime.jsonl \
  --nq-resources configs/controller/nq_train_resources_v1.json \
  --preflight --output-dir /path/to/new_preflight
```

The registry lists the paths, sizes and SHA-256 of the corpus, index, IDs, offsets, recovery metadata and full check report. Preflight does not load models and verifies all fingerprints; when no resources are specified, the old input-format pre-check can still run, but this cannot be used to claim the resources are ready. Real execution also re-verifies the resources.

`--nq-resources` and the old `--nq-index/--nq-metadata` are mutually exclusive. The old interface continues to be used for JSONL metadata containing full text; an ID-only sidecar must not be passed to the old interface. The new provider reuses the original NQFaissBackend, the original BGE query prefix, L2 normalization, IP scores and the Top20 cap, and passages passed to downstream modules contain only passage_id/text/score.

Local validation uses FAISS CPU 1.15.1 Flat read-only mmap, with BGE encoding in a separate process, avoiding the earlier macOS OpenMP conflict. The full CUDA environment still needs to be locked and measured independently; the local CPU environment cannot substitute for the GPU seal. The new provider explicitly checks mmap support; if it is missing, an error is raised, preventing a silent switch to a higher-footprint loading method.

Resource check and replay:

```bash
python scripts/check_nq_train_index_mapping.py --output /path/to/new_mapping.json
python scripts/replay_nq_controller_retrieval.py --output /path/to/new_replay.json
```

## FEVER resource locations and recovery

The two input ZIPs are `for_fever_c1.zip` (424,325,031 bytes) and `fever_corpus.zip` (1,265,030,053 bytes), sourced from the team ELEC5623 Drive folder. Checksums are in [import_validation.json](../artifacts/controller/fever_resources_v1/import_validation.json).

Unpacked locally into the new, Git-ignored `data/processed/controller_resources/fever_v1/`, without overwriting the original cloud placeholder files:

| File | Purpose |
|---|---|
| `fever_corpus.jsonl` | Original text body |
| `fever_bge_ivfpq.index` | Original IndexIVFPQ |
| `fever_bge_ivfpq_ids.jsonl` | Vector index to passage ID |
| `fever_bge_ivfpq_offsets.npy` | Vector index to original-text file position, generated this round |
| `fever_bge_ivfpq_meta.json` | Original index-build metadata, identical to the repository record |

Only the hash of the historical `fever_dev.jsonl` bundled in the ZIP is recorded; it was not installed and did not overwrite the T2 source data. Its byte size differs from the frozen source; this time no questions were reselected, no labels replaced and the held-out pool was not modified based on that copy.

```bash
python scripts/import_fever_controller_resources.py \
  --c1-zip /path/to/for_fever_c1.zip --corpus-zip /path/to/fever_corpus.zip \
  --resource-dir /path/to/new_fever_resources --report /path/to/new_import.json
python scripts/check_fever_controller_resources.py \
  --resource-dir /path/to/new_fever_resources --import-report /path/to/new_import.json \
  --output /path/to/new_structure_check.json
```

Only explicitly specified files are extracted to explicit destinations; overwrites, symbolic links and duplicate members are rejected. Fully reading ZIP members verifies CRC. Before generating offsets, all ID order, duplicate IDs, text and counts are checked. The FAISS check uses three PQ-reconstructed vectors to confirm that search results can read back the original text; this step is not used to prove BGE semantic matching or QA gains.

## Subsequent execution order

1. Add the FEVER dynamic retrieval provider, and verify compatibility of the fixed BGE query snapshot with the original retrieval results.
2. Generate original-protocol C1 Top5/answers for the frozen FEVER 900 items and seal them; keep development, selection-run and final outputs separate, and do not drop samples based on answer results.
3. Add T3 Full, No Probe, the offline human-type comparison, per-group budgets and full-sample paired acceptance; register the actual CUDA environment and create the execution seal.
4. Compare repairs, harmful repairs, routing losses and costs on the same batch of inputs, then decide on focused optimization; do not use the held-out pool for tuning in advance.

Full CPU regression: **all 706 tests passed**. Full validation and source diff registration are in [this round's acceptance](../artifacts/controller/resource_adapter_v1/README.md). The T1 protocol and the original T2 frozen records keep their original bytes; new progress is appended records and does not write back historical acceptance conclusions.
