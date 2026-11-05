# v8r3 results: format fix passes, semantic verification is still the main limitation

On 2026-10-19, Kaggle version 1 ran successfully in 553.1 seconds (about 9.2 minutes), with a fixed set of 8 development samples, 32 real outputs across four groups, and 43 model generations in total. All records passed the frozen CPU decision and cost replay, and the engineering gates recomputed locally match the cloud. **The three v8 groups had 0 format errors and the full pipeline began producing valid repairs; this is still not acceptance of a reliable net gain or of a default policy.**

## Provenance and consistency

- [Kaggle execution log](https://www.kaggle.com/code/yearo111/e-v8r3-smoke8-contract-fix-20261019/log?scriptVersionId=359908613): private Notebook, T4 ×2.
- Fixed N1/N2/N8/N12/N15/N31/N48/N96: 4 accepted correct controls and 4 internal E targets. All are previously used development samples; no blind-test or held-out claim is made.
- The model is still Qwen/Qwen2.5-7B-Instruct, revision `a09a35458c702b33eeacc393d103063234e8bc28`; NF4, greedy, seed 42 and the 384-token budget are the same as in the previous round. All semantic verification uses the same Qwen and cannot be called independent model verification.
- At runtime only the C1 question, the original answer and the actual retrieved Top-5 are used; gold and human labels are used only for offline evaluation and review. The inputs are verbatim identical to r2.
- r3 frozen bundle SHA-256: `72ed761fd8017ebbaf439729ff318caa55c0823acf4b62429c4db8f31310b6f1`.
- Result ZIP SHA-256: `766c9c377be6b7cc4123a1225f3b4670ae10348df7db3a703e72d9665d36da45`.
- For the 41 generations shared by both rounds, the decoded outputs are all verbatim identical; r3 adds 2 N96 candidate verifications. This shows that the difference in this round comes from the explicit grammar recovery and the subsequent pipeline stages being reached, with no change to the semantic prompts or the runtime reference answers.

## Engineering acceptance

The v8 stages ran 35 times in total: 16 valid original checks, 11 valid candidate generations and 5 valid candidate verifications; another 3 candidate generations were rejected by the source constraint, all in the no-original-check group. Format rejections were 0 and unclassified rejections were 0; there were 2 complete three-stage paths, 0 post-hoc truncations, and 43 total generations, below the limit of 64. The 111 portable checks passed in the cloud. The v7 control still has its 3 pre-existing invalid_output, which do not count toward the effect of the v8 grammar fix.

The r3 pre-registered engineering policy distinguishes format faults from correct source/length rejections. It did not lower the requirement for verbatim contiguous source localization, nor did it recompute r2's old gate failure as a pass. A source rejection is not a successful repair, and an engineering pass does not mean the answer is correct. The 75-sample rerun was not started automatically.

## Actual outputs

The EM below is strict reference-answer matching, with a denominator of 8 for each group; "changed answers" is not the number of successful repairs.

| Group | EM correct | Changed answers | Model calls | Key status |
|---|---:|---:|---:|---|
| v7 control | 1→2 | 5 | 8 | 3 invalid_output, 5 repaired |
| Full v8r3 | 1→1 | 1 | 13 | 5 original supported, 1 candidate unchanged, 1 candidate rejected, 1 repaired |
| No original-answer verification | 1→0 | 2 | 11 | 2 unchanged, 3 source rejections, 1 candidate rejected, 2 changed answers |
| No candidate verification | 1→2 | 2 | 11 | 5 original supported, 1 unchanged, 2 changed answers |

The full pipeline changed only N96: `Hungarian literature` → `the pastry being a finger food : the consumer would pick apart the bread as a monkey would`. Citation R1 directly explains the origin of the name; the assisted review suggests treating it as a repair, still pending human confirmation. The offline reference is `being a finger food: the consumer would pick apart the bread as a monkey would`; the candidate has an extra `the pastry`, so strict EM is still 0, and therefore overall EM did not increase. The scoring rule must not be changed post hoc for this sample.

In the full pipeline all four correct controls ultimately kept the original answer, but N12's original check still misjudged it as wrong, and it was kept only because the later candidate was identical to the original answer. Of the four E targets, N8 and N31 were misjudged as already supported and skipped repair; N48's correct candidate Govind Ballabh Pant Sagar was still wrongly rejected by the subsequent verification. These are semantic errors in matching answers to evidence, not remaining field-format faults.

## Assisted review and next round

[10 changed answers pending review](../artifacts/answer_repair/v8r3_smoke8/review/changed_outputs_pending.jsonl) correspond to 5 unique IDs; the [assisted opinion](../artifacts/answer_repair/v8r3_smoke8/review/assisted_changed_outputs.jsonl) suggests that N31/N48/N96 are repairs, N12 is an unnecessary expansion, and N15 degrades a term into a restatement of the phenomenon. Duplicates across groups cannot be treated as independent samples, and the assisted opinion does not override human conclusions. The human-review fields of all new final answers are still pending.

The next round will first freeze and run a controlled test of a verification method that "extracts facts from the evidence according to the question, then compares the answer under review", focusing on work identity, the direction of comparison relations and term type, while keeping the same correct controls, distractor evidence and cost accounting. It is an algorithmic route still to be validated and cannot be presumed to bring an improvement. Engineering is now measurable, so the main effort should shift to reducing semantic wrong acceptances and wrong rejections; once behaviour is stable, run the fixed 75-sample development set, then add natural target errors and use a common untuned test set. The current 8 samples are only for engineering debugging and are insufficient to estimate the final system gain.

## Storage locations

- [Raw outputs](../artifacts/answer_repair/v8r3_smoke8/kaggle_20261019/)
- [Frozen CPU replay](../artifacts/answer_repair/v8r3_smoke8/release_audit/cpu_replay.json)
- [Result validation and cross-version consistency](../artifacts/answer_repair/v8r3_smoke8/result_validation.json)
- [Cloud engineering gates](../artifacts/answer_repair/v8r3_smoke8/kaggle_20261019/smoke_acceptance.json)
