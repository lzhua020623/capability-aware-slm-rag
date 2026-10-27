# NQ E teacher pilot · 2026-10-28

**500 real teacher requests completed**, one response per frozen Train ID, no failures or pending requests. Independent replay verifies the original seal, exact order, maximum cost reservations, provider IDs/model version, all raw outputs, citations and local contract against the journal. This is teacher-data preparation, not a trained Qwen result or E promotion.

| Outcome | Count |
|---|---:|
| supported / insufficient / contradicted | 320 / 122 / 58 |
| keep / replace | 451 / 49 |
| unresolved | 131 |
| Target tokens with pinned Qwen EOS, min / median / max | 49 / 75 / 126 |
| Targets exceeding 384 | 0 |
| Client cost upper estimate | $6.58152 of $15 |
| Human-confirmed semantic / SFT-approved targets | 0 / 0 |

Model: Azure GPT-6.1 Sol `2026-09-29`. First 20 calls used legacy pacing; calls 21–500 used the registered token-window profile. The latter responses' provider `created` timestamp span is approximately 99.93 minutes; it excludes the initial cooldown and full final-call duration. No new model, prompt, sample or budget was substituted. Invoice cost is not verified.

The offline single-reference EM diagnostic is **95→112**, 18 fixes / 1 harm. This is a balanced 500-item Train pilot, not natural-population accuracy, student improvement or final-test evidence. EM/reference agreement does not establish an evidence-verification label; aliases and ambiguous event/entity references need semantic review.

## Files

- [Audit summary](audit_summary.json), [gate state](gate_status.json), [full audited records](audited_records.jsonl.gz). The latter contains offline reference diagnostics and is **not** an API payload or an approved SFT dataset.
- [100-item review page](review/semantic_review100.html), [review rows](review/semantic_review100.jsonl), [review manifest](review/review_manifest.json). T001–T100 include all 49 replacements, all 9 contradicted/no-candidate cases, 21 supported keeps and 21 insufficient keeps. Direct SHA256(sample_id) ordering and fixed round-robin behavior implement the frozen sampling rule without gold/score screening. All human fields start empty; browser storage is isolated by sample/output fingerprints. T011 is the EM-positive→negative Ark-location example requiring event/time review.
- [Original runtime/result bundle](teacher_runtime_results.tar.gz), [503-member byte manifest](runtime_bundle_manifest.json): original 500-task input file, original input-export manifest, full journal and 500 raw Azure responses. Explicitly excludes keys, launchers and offline reference content. The original input manifest's zero-call/pre-deployment flags are historical preparation metadata; actual execution is certified by the journal and raw responses.
- [Separate original offline references](offline_references.jsonl.gz): only for offline replay/review, never teacher context.
- [Full regression log](full_tests.log), [audit environment](audit_environment.json), [tested sources](source_snapshot.json), [post-work checks](post_selfcheck.json), [all archive hashes](checksums.json).

**1,351 tests passed in 116.778 seconds** against audit source `6e02c4c`. The browser JavaScript passes `node --check`; all 100 cards and 49 replacement-harm fields were independently counted. The pinned tokenizer's cached files match the previous frozen fingerprints, with no tokenizer/model download or model-weight loading. The old temporary validation environment had lost its venv metadata/dependencies; a fresh isolated environment restored the previously recorded package versions. Auditing added zero Azure calls. Existing data, C1, routing, E availability and the user's annotation remain unchanged.

## Replay in a fresh working copy

Restore the runtime bundle beneath `data/processed/e_distillation_v1/`, and separately decompress offline references to `pilot_20261027/offline_references.jsonl`. Use the registered dependencies and cached Qwen tokenizer revision `a09a35458c702b33eeacc393d103063234e8bc28`; model weights and an Azure key are unnecessary. Then run:

```bash
python scripts/audit_e_teacher_pilot.py --output /tmp/e-teacher-pilot-fresh-audit
```

The audit refuses an active writer and requires a fresh output directory, so existing human work is preserved. All report and file hashes are available above. The original repository split/label/ID metadata is also required by the role guard.

## Next gate

Review 100 cases: at least 95 wholly correct supervision outputs, no harmful replacements among the reviewed changes, and no unresolved judgments. The frozen [teacher registration](../../../configs/distillation/e_teacher_sol_pilot_v1.json) still disables expansion and training. Passing the computational checks does not approve any target. Only after the semantic gate can a separate expansion/training protocol admit a 5,000-example candidate pool (ceiling 10,000), define filtering/mixture/budget, and test Kaggle QLoRA for the E adapter while C1 stays fixed.

See the [Chinese result report](../../../docs/E_TEACHER_PILOT_RESULTS_20261028.md). Production E remains unavailable, and P60/shared-baseline delivery gaps remain separate tasks.
