# Phase 6-3 official GPU scoring

All 2,381 rows are scored; 2,313 ordered terminal results passed journal checksums and frozen guard validation. The 68 unchanged rows retain their original answers in both guarded and unguarded scoring. Generation failures: 0.

EM means SQuAD/NQ normalized exact match against the best reference alias; F1 means token overlap F1 against the best alias. Label=correct is an AI Diagnosis stratum, not EM correctness.

## Before / after

Values are percentages; deltas are percentage points. Brackets are 95% CIs. EM levels use Wilson; F1 levels and paired deltas use 10,000 paired row bootstrap replicates (seed 6302026). All rows, including unchanged rows, enter the bootstrap.

| Label | N | EM before [CI] | EM after [CI] | Delta EM [CI] | F1 before [CI] | F1 after [CI] | Delta F1 [CI] |
|---|---:|---:|---:|---:|---:|---:|---:|
| correct | 1772 | 48.36 [46.04, 50.69] | 47.01 [44.69, 49.34] | -1.35 [-2.60, -0.17] | 66.89 [65.09, 68.71] | 64.35 [62.44, 66.25] | -2.55 [-3.66, -1.48] |
| evidence_insufficiency | 244 | 0.00 [0.00, 1.55] | 5.74 [3.45, 9.40] | 5.74 [2.87, 8.61] | 2.11 [1.22, 3.13] | 10.46 [7.26, 13.92] | 8.35 [5.29, 11.62] |
| evidence_accessibility | 68 | 0.00 [0.00, 5.35] | 13.24 [7.12, 23.28] | 13.24 [5.88, 22.06] | 8.10 [4.26, 12.36] | 21.93 [13.64, 30.77] | 13.82 [6.29, 22.26] |
| evidence_utilisation | 297 | 0.00 [0.00, 1.28] | 12.12 [8.89, 16.32] | 12.12 [8.42, 15.82] | 6.32 [4.66, 8.06] | 20.67 [16.76, 24.65] | 14.34 [10.76, 18.18] |
| overall | 2381 | 35.99 [34.09, 37.94] | 37.46 [35.54, 39.43] | 1.47 [0.38, 2.60] | 51.02 [49.25, 52.75] | 52.17 [50.36, 53.90] | 1.14 [0.11, 2.19] |

## Guard and harm

Fixed: baseline EM=0 to EM=1. Broken: baseline EM=1 to EM=0. Harm uses the originally EM-correct rows as denominator. Unguarded uses raw generations, with unchanged/error rows retaining baseline answers.

| Label | Guarded fixed / broken | Unguarded fixed / broken | Guard rejected | Rejected raw EM=1 | Guarded harm [CI] |
|---|---:|---:|---:|---:|---:|
| correct | 49 / 73 | 52 / 102 | 277 | 16 | 73/857 = 8.52% [6.83, 10.58] |
| evidence_insufficiency | 14 / 0 | 14 / 0 | 59 | 0 | N/A (0 baseline EM-correct) |
| evidence_accessibility | 9 / 0 | 10 / 0 | 8 | 1 | N/A (0 baseline EM-correct) |
| evidence_utilisation | 36 / 0 | 38 / 0 | 64 | 2 | N/A (0 baseline EM-correct) |
| overall | 108 / 73 | 114 / 102 | 408 | 19 | 73/857 = 8.52% [6.83, 10.58] |

Guard rejection: 408/2313 = 17.64%; raw EM=1 among rejected: 19/408 = 4.66%.
Guard prevents 29 EM breaks and forgoes 6 EM fixes; net EM benefit of guard is 23 rows.

| Policy | EM after [CI] | F1 after [CI] | Delta EM [CI] | Delta F1 [CI] | Harm [CI] |
|---|---:|---:|---:|---:|---:|
| guarded | 37.46 [35.54, 39.43] | 52.17 [50.36, 53.90] | 1.47 [0.38, 2.60] | 1.14 [0.11, 2.19] | 8.52% [6.83, 10.58] |
| unguarded | 36.50 [34.59, 38.45] | 51.32 [49.52, 53.07] | 0.50 [-0.71, 1.72] | 0.30 [-0.87, 1.42] | 11.90% [9.90, 14.24] |

## Validation routing weighted estimate

Recomputed from commit `338ff34b039ed367220b0b916b48c4f7fc715e20`: NQ validation 406, excluding 1 Prompt Development row gives 405; final Controller action routes 49 to retrieval_enhancement. Label counts: correct 19, insufficiency 23, accessibility 0, utilisation 7. Weights are 19/49, 23/49, 0/49, 7/49. Source paths and SHA256 are in routing_weights.json. No test records are read.

Weighted delta = sum(label weight * label mean paired delta). CIs resample within each label and hold validation weights fixed; they do not include uncertainty in the 49-row routing composition.

| Policy | Weighted delta EM [CI] | Weighted delta F1 [CI] |
|---|---:|---:|
| guarded | 3.90 [2.41, 5.50] | 4.98 [3.39, 6.69] |
| unguarded | 3.43 [1.92, 5.04] | 4.84 [3.21, 6.58] |

## Generation latency

All 2,313 calls: median 0.700995 s, mean 0.772012 s; total 1785.663 s. First lazy call includes model load; receipt model load 12.213637 s. Excluding the first call: median 0.700984 s, mean 0.759496 s. These are worker generation-call wall times, not pure GPU kernel timings.

## Provenance and source hash acceptance

Every win2 package file was verified against its package_manifest. The evaluator retains strict hash comparison. The explicit --package-manifest option accepts only the reviewed manifest with SHA256 `3c72eb6d5ba62714711b18ae71c7a1c78943f5f18565ab6e1342005a55d79f7d`; source keys and all non-source run settings must still match. Acceptance is logged in metrics.json. The Windows lock and commit-receipt fixes change source bytes without changing prompts, generation, or the answer guard. Both fixes were copied byte-for-byte from win2. Archive and reference hashes, model identity, sessions, and source file hashes are preserved in audit.json and raw/.

## Reference comparison

Comparison to supplied quick computation (after rounding): {"insufficiency_em_0_to_5.74": true, "overall_em_35.99_to_37.46": true, "broken_73": true, "weighted_delta_em_3.90": true}

## Limitations

C1 reproduction has not been run yet under transformers 5.2.0 and Qwen commit a09a35458c702b33eeacc393d103063234e8bc28. Gains on failure-label rows may partly come from regeneration noise. Historical C1 was not revision pinned and initial answers are annotation-derived; this comparison does not isolate a causal retrieval effect. The tokenizer commit was inferred from the model's same-repository snapshot, not independently observed. Hardware: NVIDIA RTX 4000 SFF Ada Generation; torch 2.6.0+cu124; bitsandbytes 0.50.2.

The validation-weighted result assumes label-conditional effects transport to actual routed queries; it is not a measured integrated Controller gain. Labels are AI initial annotations, not independently verified human labels. CIs assume exchangeable rows, do not model question clusters, historical model drift or routing-weight uncertainty, and are descriptive intervals on a fixed development cohort. This work ran offline scoring and fake-model tests only, with no new generation or C1 reproduction.
