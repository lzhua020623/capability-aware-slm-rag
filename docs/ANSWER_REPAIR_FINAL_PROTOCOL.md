# Closing protocol for E's final experiment and review

The user confirmed execution on 2026-10-19. The final r4 smoke has completed; because the N31/N48 regression conditions failed, r3 was selected as the final frozen experiment version per the pre-set rule, and no r5 is added. After that, the strategy is no longer changed based on development or unseen results. Engineering completion, development results, unseen results and assisted/human review are reported separately; r3 was not automatically promoted to default.

## 1. Final smoke round

r4 changes only the verification prompt, FACT first and then six judgements, with the ten-line parsing and stop contract synchronised; it does not change the shared SEMANTICS, the minimal-repair prompt, the candidate grammar, contiguous source localisation, model/revision, greedy/seed, 384 tokens, the three stages or the four groups. It reuses the fixed eight development samples and the r3 engineering threshold. The full group requires in advance that for N31 the original check is unsupported; for N48 the original check is unsupported and the correct candidate is supported; for N12 the original check is supported and the original answer is kept. All other samples are saved, and EM/F1 are not used for selection. If all pass, select r4, otherwise keep r3; no r5 is added. The downstream candidate-citation failure of N31 is still recorded as is.

## 2. Final 75-item development numbers

Using the original 75 items, C1 reuses existing records, and four groups are newly generated: v7, final v8, without original check, and without candidate check. Model, evidence, scorer, gold and budget remain frozen. The same-batch v7 20→23 serves as the development reference. Record EM/F1, rejections at each stage, changed answers, extra calls, tokens and latency. Once results are obtained, stop modifying, and do not claim unseen gains.

## 3. One-off unseen stratified challenge set for E

From the internal calibration confirmed as not used for tuning, automatically draw 100 items with EM=0 whose reference passage hits the candidates, plus 25 EM=1 controls, reusing the first-batch screening and fixed ID hash ordering. No item-by-item preview or manual selection; first fix the seed, IDs, source hashes, purpose and the seen exclusion set. Automatic screening uses offline gold; runtime prompts must not use gold/EM/labels. Candidates cannot be directly regarded as utilisation; report per stratum, and do not extrapolate to the natural distribution of all of NQ.

C1 reuses the existing frozen predictions; v7 and final v8 are newly run and reported once, truthfully. Infrastructure failures may be resumed under the same fingerprint/configuration; after seeing results, it is not allowed to switch strategy, switch samples or add tuning. Calibration samples must be registered as used and isolated from Diagnosis training; the purpose and end-to-end IDs of the team's shared reserved_evaluation still need coordination, and the whole system cannot currently be called independently tested.

`scripts/prepare_answer_repair_unseen.py` must read the selected final 75-item frozen bundle and its fingerprint before it generates these 125 inputs. It re-verifies all source shards, recomputes the original split, reuses the original hash ordering, and excludes the normalised question groups containing seen IDs; additional seen IDs can be registered via `--additional-exposed-ids`. The console outputs only counts and fingerprints, without showing questions, answers or evidence. Before the final version is selected, it is only tool preparation, and real samples are not drawn in advance.

Status on 2026-10-20: after r4 failed the real regression, r3 was selected per the original rule, and the final 75-item bundle `b5be903d…` was frozen. The above 125 inputs were then frozen automatically; source fingerprints, fixed IDs and the E consumption registration are in `artifacts/answer_repair/unseen_nq125/manifest.json`; there was no item-by-item preview or manual screening. Two Kaggle jobs have been submitted: the 75-item version `360070600` and the 125-item version `360075807`. Completed results have not yet been collected, and job submission cannot be regarded as valid completed inference. Team-wide training isolation is still pending coordination.

## 4. Closing the review

The new 75 items and the unseen batch review the union of all changed answers of v7 and final v8; the same C1 and identical final answers can be checked once and reused, while answers differing by strategy are reviewed separately. The assistant assists first, and the user confirms by number; ambiguities remain unresolved, without deleting samples or changing gold. Reviewing only changed answers allows reporting the number fixed, the number of harmful repairs and the batch's net change, but not computing complete semantic accuracy or a repair rate with all originally wrong answers as the denominator. The 75 items already have initial human review labels, and their corresponding denominator can be used separately.

Historical v7 held-out reviews changed answers using a fixed sampling rule, recording strata, seed, number sampled and coverage. 841 is the number of historical prediction records and cannot be written as full human review completed. Assisted review is not independent blind labelling.

The historical sampling rule is fixed in `configs/answer_repair_v7_historical_review_v1.json`: filter only by changed answer, seed 42, stratify by cohort/dataset/strategy, take the first 20 per stratum by ID hash ordering (take all if fewer), without using EM or correctness for selection. 45 of the 77 historical held-out changed answers were selected: FEVER 5/5, old NQ 20/20, batch 4 NQ 20/52; on 2026-10-20 all assistant assisted opinions were completed, and human confirmation is still 0/45. The list, rationale, stratum statistics and read-only web page are in `artifacts/answer_repair/v7_historical_review_v1/`. Stratum coverage is reported separately, and these 45 are not treated as the known repair rate of all 77.

## 5. Report and team

Organise the E chapter around fixes, harmful repairs, the repair rate of conservative verification together with its cost, and applicable trigger conditions; conclusions follow the final actual results. The first paragraph of the handoff states the final frozen experiment version and the default strategy status. Confirm with Zheng the end-to-end executor, machine, shared test IDs and data use; E's --all component experiment cannot replace Diagnosis→Controller→E end-to-end results.

## Result intake and engineering check

On 2026-10-21 both complete exports were received; the 300/250 predictions all passed frozen replay and summary recomputation, without strategy changes or reruns. This round's GPU experiments and the assistant's assisted review are complete; human confirmation of the 42 final main-strategy changed answers and the 45 historical sampled items is still pending. Current numbers and interpretation are in the [final report](ANSWER_REPAIR_RESULTS_20261021_FINAL.md).

`scripts/collect_answer_repair_final.py` receives the corresponding Kaggle output ZIP (`--run dev75` or `--run challenge125`), verifies the CRC, frozen bundle/model fingerprints, execution environment and the original test-pass records, then uses that experiment's frozen code to replay all 300/250 results and recompute the summary. `--python` points to the same Python major/minor version as Kaggle, avoiding cross-version floating-point summation differences; the original ZIP and exported files are kept as is.

Only the changed answers of v7 and final r3 enter `review/primary_changed_pairs.jsonl`; ablation records are kept in the full audit. All final semantic judgements and human fields remain pending, and engineering replay cannot replace human review. The tool has completed integration testing with 250 independently labelled synthetic records; the default path rejects synthetic results, and even when explicitly allowed they can only be written to a temporary directory and cannot be committed as model experiment results.

Human first-batch update on 2026-10-21: F1–F42 have been reviewed, with correctness finalized for 32 and 10 unresolved; preferences do not substitute for binary wrong judgements, and accepted pending items keep null. Confirmations are stored separately in `data/annotations/human/answer_repair_final_v1_batch01_20261021.*`, and historical H45 was not automatically confirmed. See the [human report](ANSWER_REPAIR_FINAL_HUMAN_REVIEW_20261021.md).
