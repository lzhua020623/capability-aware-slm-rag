# E teacher pilot production results · 2026-10-28

All 500 NQ Train teacher tasks are complete and the independent replay passed. The teacher is Azure GPT-6.1 Sol `2026-09-29`; the current result is training-data pilot production: **Qwen has not been trained yet, E has not been promoted yet, and the final test set was not involved**.

| Check | Result |
|---|---:|
| Unique requests/raw responses/successful settlements | 500/500/500 |
| Failures/unsettled/duplicate requests | 0/0/0 |
| Model version, JSON, evidence provenance and original ledger replay | 500/500 passed |
| Student-visible JSON + EOS length | Min 49, median 75, max 126 tokens; all below 384 |
| Supports original answer/insufficient evidence/clearly contradicts original answer | 320/122/58 |
| Finally kept/replacement proposed | 451/49 |
| Unresolved | 131 = 122 insufficient + 9 contradicted but without a qualified candidate |
| Conservative client-side cost | $6.58152/$15; not a verified Azure bill |

All inputs are still the original question, the original C1 answer and the retrieved Top-5; the teacher did not receive gold or AI initial labels. The 500 items were sampled from the allowed Train at 125 for each of the four AI initial-label classes, so they do not represent the natural distribution. The 500 development/500 selection-run items of Validation and the full Test of 11,114 items were not used for this teacher generation.

## What these numbers mean

From an engineering standpoint, the SOTA teacher can stably output the current E contract: no truncation, format fallback or illegal citations, and the short targets also fit the later student 384-token budget. **Legal citations do not prove that verification labels or replacement semantics are correct, and 49 replacements are not 49 fixed.**

Using the physically isolated offline reference, for EM diagnosis only: C1 is 95/500, and after applying the teacher candidates it is 112/500, with 18 going from wrong to right and 1 from right to wrong, a net gain of 17. EM misses equivalent answers and cannot directly judge whether evidence is sufficient; these numbers cannot be written as post-distillation Qwen improvement, final test gain, or teacher semantics having passed.

Items 1–20 come from the old throttling and items 21–500 from token-window v2; all IDs, task contents, model, gate and cumulative ledger are consistent. The provider `created` timestamps of the last 480 responses span about **99.93 minutes**, which is the span of service response timestamps, excluding the initial cooldown and the full duration of the last request, and is not GPU training time. The median single service latency is about 3.43 seconds.

## The frozen 100-item semantic review

The [review web page](../artifacts/controller/e_teacher_pilot_20261028/review/semantic_review100.html) and [JSONL](../artifacts/controller/e_teacher_pilot_20261028/review/semantic_review100.jsonl) are prepared, numbered T001–T100. Each question contains the original question, C1, the teacher judgment and candidate, cited sentence numbers, the full Top-5 and the collapsed offline reference; all are unreviewed by default.

- 49 replacements: all included.
- 21 supported kept, 21 insufficient kept.
- 9 contradicted but without a candidate: all included.

Sampling is performed by the registered behavior groups, ascending by `SHA256(sample_id)` within each group, with no tunable seed; there is no filtering by gold, EM or assistant judgment. When the replacement group has fewer than 50, all are included, and the remaining slots are allocated in rotation among the three kept groups; after the rare groups are fully included, the remainder is allocated. The review counts of all three verification labels meet the requirement of at least 10.

**T011 needs focused attention on the event and time qualifiers**: the original answer is Kiriath-Jearim, and the teacher changed it to Jerusalem. The evidence describes the location of the Ark at different stages; check whether it really rules out the original answer. Currently it is only recorded as EM right→wrong and pending review, not pre-filled as a human-review harmful repair. The Texans/Chiefs entity risk from the initial connection is kept separately as an explanatory case, and is not temporarily inserted into the frozen 100-item review.

The web page allows filling in item by item and exporting JSON; you can also confirm by T number in chat. Browser records are isolated by sample and raw response hash, so a different batch of content will not reuse old selections. gold only helps understand the question and cannot replace evidence verification. The page does not call models or write training targets; exported human opinions still need to be checked on import.

## Next steps and stopping conditions

First complete the semantic confirmation of these 100 items. Per the original registration, at least 95 items with fully correct supervision semantics, zero harmful repairs among the reviewed replacements and no pending items are required to pass the teacher quality gate. The current numbers of human confirmations and of SFT approvals are both 0. Passing the length check does not equal passing semantics.

After the gate passes, separately seal the expansion list, cost, target mix and filtering method, then expand to a candidate 5,000 qualified supervision items (cap 10,000) and measure Kaggle QLoRA. Only the E adapter is trained, with C1 fixed; the training config, long-input acceptance, harmful-repair cap on correct questions and promotion rules are registered first. If not passed, record the reason; this round's gate cannot be relaxed afterwards, and the 500 items cannot be put into training wholesale.

The raw inputs/responses, full ledger, full replay, environment, fingerprints and tests are in the [results archive](../artifacts/controller/e_teacher_pilot_20261028/README.md). Integration gaps such as the P60 original features/provenance and the team common baseline are not automatically resolved by the completion of the teacher pilot production.
