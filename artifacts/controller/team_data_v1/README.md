# Team NQ alignment and boundaries · 2026-10-27

Source branch: `controller_2`; implementation: `e8c78e0d03b6d001e577943d455fb60797efe0a3`.
Team source: `feature/baseline-evaluation@387ddafc82bc939ebd27b3137d580db6ecb2e04a`, verified against the remote head after the audit.

- Official split copied byte-for-byte: 51,866 train / 11,117 validation / 11,114 test (74,097 total). No reassignment or order change.
- Local AI-label index covers the exact same 74,097 IDs; train/validation label-count metadata matches. Counts do not certify individual labels.
- All 15 historical C1 delivery shards exist locally and match the original manifest hashes. Raw historical C1 is not the new context_fallback baseline output.
- Conservative admitted pools: 46,421 train / 9,440 validation. Legacy reserve and selection restrictions are retained; validation also excludes registered downstream exposure. These are use restrictions, not new splits.
- All 55,861 admitted rows passed query/answer/Top-5/gold-reference structure checks. Cross-role normalized-query fingerprint overlap is zero. Test questions, evidence and gold were skipped by ID before JSON decoding.
- Registered downstream exposure overlaps 603 distinct official test IDs. The remaining 10,511 are outside the registered sources, not proven team-unseen. Keep the official 11,114 primary comparison, disclose overlap and add the registered-exposure-excluded comparison in the final protocol.

## Enforced current entry points

`src/data/team_nq.py` pins the original split/source; checks exact string IDs, role eligibility, parent/donor provenance and bundle hashes; partitions before negative construction; and exports a train-only teacher payload containing runtime fields. Builders write only train/validation to fresh directories. Trainers check complete bundles before model/CUDA initialization or limits, and never load or score test. Comparison caches and optional subsets cannot bypass the role check. Current production input checks run before Diagnosis/augmentation; unregistered FEVER and final-test execution fail closed. Low-level fictional engineering injections are labelled as unenforced.

Historical sealed runners/packages/results remain under their original protocols. All model assets, E unavailable modes and inference budgets are unchanged. Only the current baseline's data-boundary registration was added.

## Validation

**1261 tests passed in 99.143 seconds** against the implementation commit above. Network model downloads were disabled; benchmark inference/training was not run.

- [Full test log](full_tests.log); [94 boundary/component checks](targeted_tests.log).
- [Raw source/ID audit](audit.json) was replayed exactly with the final data reader.
- [Production readiness](readiness.json) still stops on the original 78-feature provider and P60 model-usage IDs.
- [Before](before.json), [post-work self-check](post_selfcheck.json), [source fingerprints](source_snapshot.json), [file fingerprints](checksums.json).

## Remaining limitations

AI Diagnosis labels and weak supervision are not semantic gold. Prior trained models' usage IDs remain unverified. No new comparison predictions/metrics/run_config have been delivered in the local branch. FEVER team roles require separate registration. No training, teacher API calls, GPU benchmark, dataset augmentation or default-weight promotion was performed.

See [the current boundary policy](../../../docs/TEAM_DATA_BOUNDARIES.md) and [T1–T9](../../../docs/CONTROLLER_TASKS.md).
