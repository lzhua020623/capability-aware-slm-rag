# Compatibility integration of the Diagnosis update · 2026-10-26

2026-10-27: The user has designated this upstream update as the main line going forward. The current config, real enhancement adapter and undelivered dependencies are in [new baseline registration](CONTROLLER_1024_BASELINE.md). The compatibility merge scope of October 26 is kept below; the historical T4/T5 still use the original frozen entry point.

Source: `feature/diagnosis-probe-zheng`, commit `138d52aa22e0eee78637e01f981a030d7ad388df`. This merge brings in the code framework and model assets, keeping the existing default experiment executor; P60 is not enabled, and no new quality experiment was run.

## Actual integration

- The original Diagnosis weights, 27-dim features, Probe prompts and Safe Binary rules are unchanged.
- Added `primary_failure_type / active_failures / failure_scores`. Active types come from a representation rule of original Stage-2 conditional probability ≥1/3, and actions are still selected by a single primary type; no independent multi-label model was trained, so multi-label recognition cannot be claimed to pass on this basis.
- Added the Enhancement request/result protocol, registered executors and an injectable Adaptive RAG host. Unregistered modules and exceptions keep the original state; success only means execution completed.
- The P60 selector, fixed config and three HGB assets are kept as an explicit optional entry point, not passed into the current `IntegratedController`. The new host's Probe-OFF does not change the Probe path of the default T4/T5.

## Compatibility and return checks

The original reranking `RefinementConfig / RefinementResult / RefinementStatus` coexists with the new Enhancement protocol; the package level keeps lazy exports and does not introduce forced Diagnosis/Probe loading for the retrieval module.

New executor checks: the returned action must match the requested action; changed_answer must match the actual string change; non-success results must not publish new answers or evidence. Violations are recorded as error and the request's incoming state is kept. The latency of new contracts must be finite and non-negative.

The official serialization entry point is each type's `to_dict()`; new decisions include a read-only score dictionary and do not rely on the generic `dataclasses.asdict()`.

## Enabling conditions P60 does not yet meet

1. Provide the training, tuning, calibration and error-analysis IDs of EXP-P60, and check the exposure of selection/final and the full held-out pool.
2. Integrate a real online feature provider. The config requires 78 features in exact order, whose names have only 75 unique values; do not deduplicate, zero-fill or use offline gold/labels on your own. Newly added features and feature computation before no_action must also be metered.
3. Use adapters for the real three enhancements; align budgets, failed calls, tokens, context and trajectories, rather than using mocks as production modules.
4. If P60 is enabled, first register it as an independent experiment, stating the attribution limits of the simultaneous changes in three-model selection, features and Probe-OFF; do not silently mix it with the single NQ threshold candidate.

P60 still directly keeps questions that Diagnosis judges correct, and will not automatically resolve these missed detections. The zero FEVER E routing and the type-discrimination problem of the old model still need to be measured, and cannot be inferred as resolved from interface tests.

## Validation scope

The merge regression includes the old executor, r2 and newly added upstream tests; additionally, the Diagnosis and Probe inputs of 800 Adaptive/No Probe items are rebuilt from the T4 ZIP and compared field by field against the original routing. This replay does not regenerate answers, does not read gold, and is not a P60 effect or final evaluation.

The acceptance receipt is in [merge self-check](../artifacts/controller/diagnosis_merge_v1/merge_self_check.json). Subsequent r2 packages must reference the fixed post-merge source; the raw bytes and original results of old run packages are unchanged. Current tasks are in [T1–T9](CONTROLLER_TASKS.md).
