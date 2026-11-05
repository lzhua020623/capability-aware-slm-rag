# NQ Train retrieval resource recovery · 2026-10-24

**The original corpus and original index have been retrieved, IDs/offsets have been recovered, and 64 independent BGE vector comparisons passed.** The source is still the team's existing NQ Train subset and its accompanying retrieval store; the full official NQ was not downloaded, and not all vectors were recomputed.

| Check | Result |
|---|---|
| corpus / FAISS vector count | 1,224,119 passages; not the number of question samples |
| Index | IndexFlatIP, 768 dimensions, all vectors finite and L2-normalized |
| IDs and offsets | Generated; original order preserved, 64 source-text locations passed |
| Sampled vector comparison | 64/64 passed; minimum cosine 0.9999998808, maximum component difference 3.39×10⁻⁷ |
| BGE validation snapshot | `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`, CPU; does not pose as the historical C1 revision |
| Latest regression | **681 tests passed**, including 16 new recovery tests |
| Original data freeze check | 552 source hash checks passed; samples/held-out pool unchanged |

The generated files are in the repository's local `data/processed/indexes/`, ignored by Git. The portable [nq_train_sidecars.tar.gz](nq_train_sidecars.tar.gz) is about 6.56 MB and contains only the three recovered sidecar files; it does not contain the original corpus, the original index or model weights. The package members and their content SHAs have all been verified.

When migrating to a new runtime environment, first obtain the original corpus/index with the same fingerprints, then extract the sidecar package. The following command refuses to overwrite existing files:

```bash
mkdir -p data/processed/indexes
tar -xkzf artifacts/controller/nq_index_recovery_v1/nq_train_sidecars.tar.gz -C data/processed/indexes
```

Key records:

- `recovered_metadata.json`: copy of the official recovered metadata, recording input/artifact SHAs, the read mode and the validation values.
- `validation.json`: run commands, CPU environment, test results, package fingerprints and the conditions still to be completed.
- `resource_discovery.json`: Drive sources and original file fingerprints.
- `crash_diagnosis.json`: minimized macOS crash summary; the original full system report is not committed.
- `recovery_mmap.log`: the successful actual recovery log.
- `recovery.log`: failure log of the initial merge process that exited with 139; the crash stack is in libomp.
- `recovery_isolated.log`: an attempt from before the application was interrupted; exit status unknown, recorded neither as a success nor as the same crash.
- `structural_diagnostic.log` / `structural_diagnostic_metadata.json`: diagnostic records of the structure-only recovery; cannot be treated as passing the independent vector comparison.
- `tests.log` / `targeted_tests.log`: full and targeted tests for the current version.
- `tests_before_runtime_fix.log`: historical regression record from before the run method was corrected.
- `data_freeze_validation.json`: the frozen T2 artifacts and sources remain consistent; its description of resource gaps is still the original frozen state.

This recovery was completed with read-only mmap and an independent single-threaded CPU encoding process, without requiring a CPU/GPU change; the original crash report does not support attributing the fault directly to memory exhaustion. No Qwen QA was run and no new accuracy was computed. Passing the sample does not mean a full passage-by-passage re-encoding verification, nor that the complete knowledge base has been manually cleaned.

FEVER package contents, NQ Train Controller provider adaptation, the new FEVER C1 and formal GPU execution acceptance are still pending; `full_T2_ready` remains false. For details see [Recovery notes](../../../docs/NQ_TRAIN_INDEX_RECOVERY.md).
