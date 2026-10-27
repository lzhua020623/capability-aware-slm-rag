# E distillation preparation · 2026-10-27

Branch: `controller_2`. This is offline pilot preparation, not an API run, a trained adapter or an E promotion. Current E remains unavailable.

## Component check

[Component audit](components.json) records the refreshed upstream heads and current production precheck. Latest Diagnosis, retrieval, reranking and E branch heads are in current ancestry. Integration compatibility changes are retained. The baseline-evaluation branch is not fully merged: its exact NQ split is already independently registered.

No upstream/fork updates were found. P60 still lacks its real 78-feature provider and model-usage IDs; production readiness is false. New trained reranker/selector assets have not passed promotion. These do not prevent independent teacher-pilot preparation, but do prevent claiming a fully runnable new end-to-end baseline.

## Prepared inputs

- [Sample manifest](sample_manifest.json): 500 train pilot IDs, balanced 125 per AI stratum; 500 development and 500 selection IDs sampled uniformly from allowed validation, mutually disjoint. AI strata are not semantic gold. No final test ID is admitted.
- [Pilot file manifest](pilot_inputs_manifest.json): actual local 500-task export, original 15 delivery shard fingerprints, and physically separate offline reference hashes. Full content stays in ignored `data/processed/e_distillation_v1/pilot_20261027/`.
- [Independent replay](pilot_independent_replay.json): every task's runtime, prompt, schema and fingerprints reconstructed exactly; no validation/test runtime export, API calls or training.
- [Protocol](../../../configs/distillation/e_prep_v1.json), [implementation and next steps](../../../docs/E_DISTILLATION_V1.md).

Teacher visible outputs are not generated yet. Deployment/version, total spend/token budget and semantic acceptance remain pending. Tokenizer length checks, GPU training preflight, numeric promotion gates and fixture seals must precede SFT. No output can be auto-approved solely by schema/citation checks.

## Verification

- [108 component preflight checks](component_preflight_tests.log).
- [20 preparation checks](preparation_tests.log), including role isolation, offline-gold stripping, exact task fingerprints, parser rejection and repeatable manifests.
- **1281 tests passed in 118.292 seconds** against `2b74628122171d818138e6123e09f71611cfefeb`; see [full log](full_tests.log), [source fingerprints](source_snapshot.json), [post-work self-check](post_selfcheck.json) and [file fingerprints](checksums.json).

The [pre-work snapshot](before.json) and the final self-check preserve the user's untracked annotation and original AGENTS.md. All old experiment packages/results, production configuration and v2a prompt/parser assets remain unchanged.

Local pinned-tokenizer audit: 500 student prompts, median 1,965 tokens, p95 2,533, maximum 3,040. These are student input counts, not teacher API cost estimates. Output-target lengths remain unchecked because no teacher targets exist. Local machine has no CUDA; actual training needs a GPU preflight. See [environment audit](local_training_environment.json).

Selection IDs are reserved for one confirmation; do not use them for prompt, routing or other-module development before that confirmation. Cross-component usage and the final execution gates still need sealing.
