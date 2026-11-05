# 10-24 Diagnosis Current Mainline Integration · 2026-10-27

The source is `feature/diagnosis-probe-zheng@138d52aa22e0eee78637e01f981a030d7ad388df`. The user asked that it be the main basis going forward; that commit is already an ancestor of controller, so no repeated merge is needed. This round adds the current baseline registration, an adapter from P64 to the real enhancements, and a strict startup entry point; no new GPU runs, training, threshold re-selection, or final evaluation were performed.

## Branch Material Check

[upstream_material_audit.json](upstream_material_audit.json) comes from the full tarball of the specified GitHub commit and an untruncated recursive tree, totaling **142 files, 0 Notebooks**. A local git grep cross-check was also done. [upstream_history_audit.json](upstream_history_audit.json) covers all **10 commits** of that branch and the change list of each.

| Material | Present in branch | Check result |
|---|---|---|
| P60 config, thresholds, and feature order | Yes | 78 positions, 75 unique names |
| Three HGB models | Yes | All 78-dimensional; local bytes identical to the GitHub raw blob |
| P64 host, Controller, enhancement protocol | Yes | Reused in this round, connected to the real enhancement code |
| Original 78-dimensional online feature implementation | Not found | `features_for` is defined only in the Protocol of `src/pipeline/adaptive_rag.py` and in test Stubs |
| P60 training/tuning/calibration/error-analysis IDs | Not found | Two old Diagnosis splits exist but cannot be treated as the source of the three HGBs; the models have no sample ID attribute |
| EXP-P26/EXP-P60 training and feature experiment scripts, Notebooks | Not found | No additional deliverables seen in the full file tree or the historical change lists |

This covers only what is in Git; it does not mean the authors have none locally or on Drive. All three keyword searches via the Drive connection returned `ACCESS_TOKEN_SCOPE_INSUFFICIENT`; no message was sent to teammates. Source and permission status are in [source_receipt.json](source_receipt.json).

## Completed Engineering Integration

The current registration is `configs/controller/current_baseline.json`. The production entry point is `src.pipeline.current.load_current_pipeline()`, with the flow C1 → Diagnosis → P60 → Controller → EnhancementExecutor, Probe OFF. The three real actions use existing modules and CallLog; the old `.run()` is not called to do the old routing again.

E is still unavailable: integrating this action does not mean it passed repair acceptance; the original answer is kept and E calls are zero. Features must be provided in the 78 original positions, with no zero-padding, no deduplication, and no approximate substitution. Production cost credentials must be bound to the current input; model-usage IDs are checked against the full held-out pool and selection by original records.

[readiness.json](readiness.json) explicitly lists two original delivery gaps; the check command stops normally with exit code 2, **does not claim that P60 real runs are ready**, and does not fall back to the old routing. This state is missing input, not a model effectiveness failure.

The adapter tests newly built this round use real module logic, real frozen HGB loading, and explicitly fictional low-level outputs; they verify only engineering behavior and produce no benchmark accuracy. The 17 new targeted tests passed, and the full 982-test regression all passed (452.020 seconds). A validation process that stalled while reading the local cache was stopped separately and then run fully with an independent cache; no tests were skipped or rewritten. Full records and before/after self-checks are in [self_check.json](self_check.json).

Next, the Diagnosis owner needs to deliver the original feature generation implementation (including required models/versions) and the usage IDs of the three HGBs. After integrating and checking them, a new protocol and independent validation will be set up separately; the already-exposed old T5 can serve only as historical/engineering material. For the detailed delivery format, see the [current mainline handover](../../../docs/CONTROLLER_1024_BASELINE.md).
