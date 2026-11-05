# E final frozen experiment results

On 2026-10-21 two complete Kaggle exports were collected; all 550 new predictions passed their respective frozen-code CPU replays and summary recomputation. The final version was selected as **v8r3** per the pre-run rules; r4's N31/N48 regressions failed, no r5 is added, and the version is not switched based on this round's numbers. **r3 is not promoted to the default policy: it protects original answers but almost never repairs, and the unseen batch had no answer changes at all.**

## Automatic metrics and cost

"Repaired/broken" below only means frozen EM went 0→1/1→0, and is not a human semantic judgement. C1 reuses the original predictions with no new calls. All new groups use the same Qwen2.5-7B-Instruct revision, NF4, greedy, seed 42 and 384 tokens per stage; runtime gold was not accessed. Latency is the cumulative model generation-call time, excluding download, loading and tests, and is not end-to-end service latency.

| Batch/group | EM correct | EM repaired/broken | Changed answers | Extra calls | Generation latency per question | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Dev75 · C1 | 20/75 | — | — | 0 | — | 0.58383 |
| Dev75 · v7 | 23/75 | 5 / 2 | 14 | 75 | 6.29 s | 0.61865 |
| Dev75 · full r3 | 20/75 | 0 / 0 | 1 | 83 | 9.70 s | 0.59658 |
| Dev75 · no original-answer check | 19/75 | 2 / 3 | 12 | 89 | 7.88 s | 0.56728 |
| Dev75 · no candidate check | 21/75 | 1 / 0 | 2 | 81 | 9.48 s | 0.60991 |
| Challenge125 · C1 | 25/125 | — | — | 0 | — | 0.46862 |
| Challenge125 · v7 | 28/125 | 5 / 2 | 27 | 125 | 5.81 s | 0.46492 |
| Challenge125 · full r3 | 25/125 | 0 / 0 | 0 | 136 | 9.25 s | 0.46862 |

The initial human labels of Dev75 are 11 error targets and 64 correct controls, which is not the same measure as EM's 20 correct. This is a repeatedly used development set, and this round's numbers cannot be used to advertise unseen gains.

Challenge125 comes from internal calibration data not used for E tuning; per the original rules it automatically took 100 candidates with EM=0 whose reference passage was hit plus 25 EM=1 controls, with no manual question picking. In the candidate stratum v7's EM went 0→5 and in the control stratum 25→23; for r3 these were 0→0 and 25→25 respectively. Screening used EM, which does not mean all 100 are true utilisation errors. The batch proportions are fixed artificially and cannot be extrapolated to the natural NQ distribution; isolation from the team's Diagnosis training and page/semantic near-duplicate isolation have not yet been established, so it can only be called an E component stratified challenge, not an independent full-system test. The paired exact test on v7's whole-batch EM gives p=0.453125, which also does not support a claim of significant gain.

## Why r3 almost never changes answers

In Dev75, 45 original answers were labelled supported, 24 original verifications were invalid, and the remaining 6 entered minimal repair; only 2 produced valid candidates, of which 1 was rejected by candidate verification, leaving 1 final changed answer. By the already confirmed initial human labels, of the 11 originally wrong answers, 6 were labelled supported, 2 still output the original answer in the repair stage, 1 had an invalid original verification, 1 had its candidate rejected, and 1 was changed. The obstacles therefore come simultaneously from semantic-verification wrong acceptances, missed repairs by candidates, wrong rejections of candidates, and citation generation failures.

In Challenge125, 75 original answers were labelled supported, 39 original verifications were invalid, 1 was uncertain, and only 10 entered minimal repair. Of these, 1 was insufficient, 6 output unchanged, 2 had invalid candidate citations, and the only 1 valid candidate was then rejected. In the end there were no changed answers at all. The 57 supported in the candidate stratum cannot all be called misjudgements, because some of them are original answers that are semantically correct but do not match on EM.

Across the full groups of both batches, invalid original verifications total 63/200: 59 were not verbatim source fragments and 4 exceeded the citation length constraint; there were no nine-line format parsing errors. Keeping the original answer when invalid conforms to the frozen safety policy, but it also consumes calls without producing repairs. Relaxing parsing might increase the number of accepted citations, but it cannot automatically solve the model's confusion of relations such as similar titles, role/actor, and definition/term. This round does not adjust these rules or rerun.

## Frozen assisted draft (before human confirmation)

The review union contains 42 results from 41 samples: 14 from dev v7, 1 from dev r3, 27 from challenge v7; challenge r3 has no changed answers. Identical sample_id cannot be deduplicated directly: F13 and F15 are two different answers to the same question and are kept separately. The 14 extra changed answers from the ablations remain in the full audit and are not mixed into the main-policy review scope.

| Stratum/policy | Suggested semantic repair | Suspected harmful repair | Correct kept/synonymous rewrite | Correctness pending |
|---|---:|---:|---:|---:|
| Dev v7 · 14 | 3 | 2 | 8 | 1 |
| Dev r3 · 1 | 1 | 0 | 0 | 0 |
| Challenge v7 · 27 | 2 | 4 | 15 | 6 |

The above is a reference-visible assistant draft frozen before human confirmation, when human confirmation was 0/42; the current human results are in the next section, and the two are kept separately, without overwriting the draft. The only changed answer in dev r3 is N96 (F15), changed from Hungarian literature to the naming origin of pulling it apart by hand like a monkey to eat it; the assisted judgement leans towards a valid repair, even though EM is still 0. The two EM control drops in challenge v7 are merely a station name changed to an address and a state name changed to a city under its jurisdiction, and the assisted judgement saw no semantic harm; however, there were also harmful repairs not fully reflected by EM, such as answering the reason for an award with "the number of awards" and answering a person with "an event".

Key ambiguities in the draft were F4, F19, F21, F22, F35, F36, F40; the unresolved scope after this confirmation is in the next section. F19's reference reverses the direction of the renaming and the evidence has two dates; F30 distinguishes a diploma from a degree; F38 has multiple return plot events and cannot be forcibly judged wrong against a single reference. The references and original scores are both kept unchanged; only review notes are recorded.

Materials: [final 42-item web page](../artifacts/answer_repair/final_review_v1/F42_review.html), [stratified assisted statistics](../artifacts/answer_repair/final_review_v1/assisted_summary.json). The historical fixed spot review of 45 items has also completed assisted review, with human 0/45; see [historical 45-item web page](../artifacts/answer_repair/v7_historical_review_v1/H45_review.html). Historical and final lists are counted separately, and the 841 historical predictions are not described as all having completed human review.

## Human confirmation update (as of 2026-10-22)

The user has gone through all of F1–F42: as of the follow-up clarification on 2026-10-22, 42 items are processed, of which 37 can be scored in binary and 5 retain scoring notes, and they are no longer treated as pending. For F12 a follow-up requirement was to follow the C1 original answer, continuing to record keep/prefer the original answer. F12/F23/F25/F36 are recorded as keeping the original answer, and the correctness of the changed answer is not forcibly scored. F21 original correct changed wrong, F22 both answers correct, F35 original wrong changed correct, F40 both answers wrong; after clarification, F37 follows the requirement of the specific recipient Continental Army and is recorded as both answers wrong, no longer counted as a harmful repair. The original gold, EM and frozen outputs are unchanged.

| Stratum/policy | Confirmed repair | Confirmed harmful repair | Correct still correct | Both answers wrong | Correctness pending |
|---|---:|---:|---:|---:|---:|
| Dev v7 · 14 | 4 | 2 | 7 | 0 | 1 |
| Dev r3 · 1 | 1 | 0 | 0 | 0 | 0 |
| Challenge v7 · 27 | 3 | 4 | 14 | 2 | 4 |

Including unresolved items, the net change across all changed answers of dev v7 can only be bounded within [+1,+2], and challenge v7 within [−5,−1]; these are not confidence intervals, and the finalized differences alone cannot be used to advertise the final gain. Full r3 dev N96 is a confirmed repair; under the existing initial human-review measure, the semantically correct count goes 64→65/75; challenge had 0 changed answers, semantic net change 0. The dev improvement cannot be generalized to unseen gains, and the default-policy status is unchanged.

F12, F23, F25 and F36 have been decided to keep C1, and F19 has completed external checking; these 5 only retain binary scoring notes and are no longer listed as to-dos, and the original human labels and net-change ranges are unchanged. Historical H1–H45 are still not human-confirmed. The current material is the [human record web page](../artifacts/answer_repair/final_review_v1/F42_human_review.html), and the detailed measures and traceability are in [this batch's human report](ANSWER_REPAIR_FINAL_HUMAN_REVIEW_20261021.md).

## Conclusions and remaining work

Current evidence supports "repair capability exists; harmful repairs and over-conservatism are the main obstacles", and does not support "the added verification has already brought a reliable net improvement". Adding samples cannot directly solve the observed relation-judgement and citation-generation problems; this round already has an extra 125 items checking generalization, so the next step prioritizes completing human confirmation and honest reporting, with no further policy tuning. Any new model, training or rules should be set up as a separate follow-up experiment with new unseen data, and this batch must not continue to be used for picking versions.

E's code for this round, component GPU experiments, frozen replay and assisted review are complete; the final 42 items are processed (37 binary-scorable, 5 retaining scoring notes); confirmation of the historical 45 items and team end-to-end acceptance are still pending. After human confirmation, compute the semantic net change of changed answers per policy and batch, keeping ambiguities; full semantic accuracy cannot be inferred from reviewing only changed answers. Dev75 can use the existing initial human-review denominator; Challenge125 has no complete initial human-review denominator. After confirming with Zheng the end-to-end executor, machine, common IDs, Diagnosis training exclusion and data roles, compare gains and costs under actual routing triggers with the frozen policy; no team message has been sent, and no claim is made that this coordination is complete.

## Traceability and verification

| Item | Dev75 | Challenge125 |
|---|---|---|
| Kaggle scriptVersionId | 360070600 | 360075807 |
| Input bundle SHA-256 | `b5be903db6efecb5681498f0110ebed65920369ca1e5d466492ebae7455fc98b` | `17dcadf90d30e8fea7a4f5d0fd259f17d9c5739484eec52d7586a5855f52855f` |
| Output ZIP SHA-256 | `15a5c987c38c0d1514af648ac190788ed4075115bf525eefbb84b3a0312f2802` | `f604e0b85266abd86efed22915a745eeb854d1bc15046ff42ea7e6856a5744eb` |
| Raw export file count | 29 | 19 |
| Frozen replay record count | 300 | 250 |

Raw ZIPs, logs, environment, predictions, summaries, replay receipts and per-file SHA-256 are stored in `artifacts/answer_repair/final_nq75/` and `unseen_nq125/`. The directory name `kaggle_20261020` corresponds to the original frozen run batch, and 2026-10-21 is the date of this receipt and organization. The original inference source code of both bundles is identical to the selected r3; the subsequent receipt, review and reporting tools do not change the frozen inference.
