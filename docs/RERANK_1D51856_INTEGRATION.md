# Reranking branch 1d51856 compatible integration · 2026-10-27

After the first commit of E distillation preparation, a final refresh found three new remote commits `12aa2c6`, `56d952f`, `1d518562daa97ad555657a1965db4b8e02815c69`. This round merges them into the current `controller_2`, keeping the upstream's original history and results; this is not an effect promotion of new weights or new paths.

## What was added

- Answer candidate deduplication and selector two-candidate/multi-candidate offline evaluation tools.
- Upstream selector training summary and evaluation summary on 2,367 historical answers.
- `IndexCandidateProvider` and the optional `--top-n` experiment: fetch deeper candidates from the original retrieval corpus, merge them with the original Top-5, then rerank. The default still uses only the original Top-5.

In the historical summary, selector multi-candidate EM is 14.07% and the reranking-only group is 11.07%, but multi-candidate uses multiple sets of already generated answers, with different cost and input conditions. Two-candidate selector is 9.97%, lower than the reranking group. The weights and the actual training/tuning IDs were not delivered with the summary, so they cannot be directly registered as a new default model, nor used as an E distillation effect.

## Integration fixes

1. The new selector evaluation first checks all IDs of every input arm before decoding candidates/references; only the allowed Train/Validation may be used, and Test cannot be hidden via limit. All arms must have the same cohort, original answers and references.
2. When reading questions from source data, raw bytes are first filtered by the required IDs, and unrequested Validation and Test are not decoded; references are aligned with the original source, and the model receives only the question, evidence and candidate answers.
3. New outputs go into the team_nq_v1 directory, refusing to overwrite historical results; the actual sample_ids, data boundaries and the not-yet-verified status of the weight provenance are recorded. The new local default model path is a separate team candidate and does not change production weights.
4. Before importing native/model libraries, the real loader for the deep candidate pool verifies the existing retrieval registration, full-file fingerprints, full mapping audit and paths, and pins the BGE revision; after loading, it verifies the FAISS type, dimension and count. Independent fabricated tests can still inject a lightweight fake index.
5. `--top-n` must be positive; the existing frozen backend has no variable pool wired in, so it is explicitly rejected, and the actual Top-5 is no longer misreported as Top-N.

The current Controller does not enable candidate_provider, and the extra retrieval is not wired into the production meter. This path needs separate real resource/GPU, memory and metering acceptance; the timing of a standalone script cannot replace full-system AC5. Default BGE-large, Top-5, call budget, P60 configuration and E unavailable are all unchanged.

## Impact on E distillation

The 500 teacher pilot-production tasks, 500 development IDs, 500 selection-run IDs and the existing v2a contract are unchanged. The new reranking tools must not use E's Selection list for development. The old [E preparation archive](../artifacts/controller/e_distillation_v1/README.md) keeps its test commit and the record of 1,281 regression tests; the final tests and source fingerprints of this update are saved separately in the [integration archive](../artifacts/controller/rerank_1d51856_v1/README.md).

The two P60 raw materials are still not complete; new baseline per-question outputs are still to be delivered. Therefore the CPU integration check of the latest source code does not equal end-to-end production readiness, nor does it change the E r2 failure conclusion.
