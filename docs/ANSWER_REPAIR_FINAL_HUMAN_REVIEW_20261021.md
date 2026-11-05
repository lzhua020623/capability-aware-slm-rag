# Human confirmation of E's final changed answers: cumulative record

First confirmed on 2026-10-21, with clarifications added on 2026-10-22. **42 items processed, 0 pending processing; 37 can be scored in binary, and 5 are processed with scoring notes retained.** This is a reference-visible chat-assisted confirmation, not an independent blind label. The original questions, gold, EM, model outputs and the assistant's initial drafts were not modified; historical H1–H45 are outside the scope of this confirmation.

The latest entry point is the [human review web page](../artifacts/answer_repair/final_review_v1/F42_human_review.html). The frozen assisted web page remains the original presentation material and keeps its snapshot of "human 0/42" from that time, which does not represent the current status.

## Changes made explicit in this round

| No. | Human decision | Effect on counts |
|---|---|---|
| F4 | Follow-up reply "F4 support", accepting in-plot relative time; the original answer Han is still wrong, the changed answer is correct | Changed from pending to fixed |
| F12 | The original full-sentence answer is better; follow-up "follow the C1 original answer"; correctness of the short answer is still pending | Not directly counted as a harmful repair |
| F21 | Follow-up confirms originally correct, changed to wrong: a general definition cannot be replaced by a severe form | Counted as a harmful repair |
| F22 | Follow-up confirms the original answer parabolic is acceptable, and the changed answer is also correct | Counted as correct and still correct |
| F23 | Again requested keeping the C1 original answer; correctness of the changed answer is pending | Not directly counted as a harmful repair |
| F25 | Again requested keeping the C1 original answer Xerxes I; correctness of the changed answer is pending | Not directly counted as a harmful repair |
| F35 | Follow-up confirms originally wrong, changed to correct: the special-case inorganic electron donor is insufficient, the general donor answer is correct | Counted as fixed |
| F36 | Requested keeping the original answer first season; correctness of the year-range changed answer is still pending | Original-answer preference recorded |
| F40 | Follow-up confirms that neither the original nor the changed answer answers the type of learning | Counted as both answers wrong |
| F37 | The question asks for the recipient of help; the user required explicitly Continental Army, Americans is not specific enough, American Revolution is an event, both answers wrong | Changed from the assistant's initial draft of suspected harmful repair to originally wrong and still wrong |

For F37 the user initially mentioned "should write the helping action"; after clarification, the criterion of the **specific recipient of help** was explicitly adopted, and who was not changed to how. The preferences for F12/F23/F25/F36 are recorded separately from binary correctness and are not generalised into global completeness or person-name rules. The supportiveness scores of the four items F12/F23/F25/F37 were not separately confirmed and remain null; the assistant's evidence judgements are still in the original draft for comparison.

The remaining 38 items of the first batch were recorded per the assistant fields the user agreed to, with existing nulls kept; the second batch resolved F4, and the third batch clarified F21/F22/F35/F40 and added the F36 original-answer preference. **Agreeing with "pending" does not automatically turn it into correct or wrong.** Results with all four fields finalized total 36; among the 37 with correctness finalized, F37's supportiveness is still not separately confirmed, which does not affect its "both answers wrong" correctness classification.

## Human results by batch and strategy

| Batch / strategy | Total changed answers | Confirmed fixed | Confirmed harmful repair | Correct and still correct | Wrong and still wrong | Correctness pending |
|---|---:|---:|---:|---:|---:|---:|
| Dev75 · v7 | 14 | 4 | 2 | 7 | 0 | 1 |
| Dev75 · full r3 | 1 | 1 | 0 | 0 | 0 | 0 |
| Challenge125 · v7 | 27 | 3 | 4 | 14 | 2 | 4 |
| Challenge125 · full r3 | 0 | 0 | 0 | 0 | 0 | 0 |

Dev v7 confirmed fixes are F4, F5, F7, F13, and harmful repairs F2, F3; full r3 confirmed fix is F15. F13 and F15 are two different outputs for the same N96, each belonging to its own strategy, and cannot be added up as two independent samples. Challenge v7 confirmed fixes are F16, F32, F35, and harmful repairs F21, F28, F30, F34; F37/F40 both answers wrong, F22 both answers correct.

Looking only at finalized items, dev v7 is +2 and challenge v7 is −1, but both groups still have pending items, **so these two differences are not the final net gain for the whole group**. The logical net-change range over all dev v7 changed answers is [+1,+2] items; for challenge v7 it is [−5,−1] items. They are the possible ranges of the unresolved binary judgements, not statistical confidence intervals. Full r3's dev net repair of +1 is finalized, and challenge has 0 changed answers, so the semantic net change of the answers is 0.

Dev75 already has initial human review with 64 correct and 11 wrong; full r3 changed only N96, now confirmed fixed, and the other 74 answers stay as they were. So under this batch's established human-review criteria, the number semantically correct can be recorded as **64→65/75**; it is still a seen development result. That question's EM is still 0, and frozen EM remains 20→20. This development semantic improvement cannot be generalised into an unseen gain, and r3 is still not promoted to default. Challenge125 has no full initial human correctness denominator, so no complete semantic accuracy or complete repair recall is computed.

## Processed with scoring notes retained

**F12, F19, F23, F25, F36.**

- F12: The full C1 original answer was kept per the user's decision; the changed answer was not scored in binary and is no longer a to-do.
- F19: External verification is complete; official sources support 6 September 1991; the assisted conclusion is that both answers are correct. The external conclusion is stored separately from the human gold standard, and the date conflict in the fixed Top-5 is still recorded; no longer a to-do.
- F23: The C1 original answer containing greyouts was kept per the user's decision; the changed answer was not scored in binary and is no longer a to-do.
- F25: The C1 original answer Xerxes I was kept per the user's decision; the changed answer was not scored in binary and is no longer a to-do.
- F36: first season was kept per the user's decision; the changed answer was not scored in binary and is no longer a to-do.

All 42 items on this page are processed, and these 5 will not be asked about again. Processing completion and binary scoring are recorded separately: a retained preference is not automatically turned into a harmful repair, nor is the assistant's external verification turned into a human gold standard; the existing counts and net-change ranges remain unchanged. Human confirmation of historical H1–H45 is still pending.

## F19 near-duplicate evidence check

The user asked why the evidence cited by F19 looks exactly the same. Passage-by-passage check: R2/R3 differ only in the leading zeros of two dates (1/01, 7/07); R4 has the same body text as R2 plus one extra sentence on the relative direction and distance from Moscow. The three passage_ids are different, and they cannot be regarded as three independent sources. Corpus construction deduplicated only by exact full text, and the retrieval check ensures passage_id uniqueness; no diversity selection was applied to near-duplicate content.

The upstream delivery shard SHA has been re-verified; the five passage IDs and original texts are consistent item by item from original shard → frozen C1 → run input → review material, so the duplication was not introduced by the review web page. The actual citation is a verbatim span of R1, and being identical to R1's corresponding content is the normal result of source localisation. The original corpus document metadata is unavailable locally, and the specific page/version source has not been verified.

R1 says 6 September, R2–R4 say 7 September; the conflict cannot be resolved by the number of duplicated passages. Subsequently the Presidential Library of Russia and Saint Petersburg city government records were checked and support 6 September; the F19 external verification task is complete, and the assisted conclusion is archived separately. See the [F19 source audit](../artifacts/answer_repair/final_review_v1/F19_source_audit_20261021.json). The frozen evidence was not deduplicated, replaced or rewritten, and gold and model results were not changed.

## Traceability and protection

The current cumulative verbatim statements, clarifications and source SHAs are in the [batch 4 confirmation input](../data/annotations/human/answer_repair_final_v1_batch04_20261022.confirmations.json); per-result judgements are in the [current human record](../data/annotations/human/answer_repair_final_v1_batch04_20261022.jsonl); stratification and unresolved items are in the [current statistics receipt](../data/annotations/human/answer_repair_final_v1_batch04_20261022.summary.json). Batch 4 links to batch 3, batch 2 and the first batch via SHA, and the old confirmations, records and statistics keep their original bytes. This round's "agree" originally corresponded to pending items, so the four missing binary fields were asked about; the user replied "yes", which was persisted per the shorthand in the question "21 changed wrong, 22 originally correct, 35 originally wrong, 40 changed wrong" and restated in the commentary. Keeping the original answer is a human choice and does not retroactively change the frozen model output. Batch 4 corrected the workflow status per the user's remark "weren't these already processed": 4 items keep the original answer and 1 completed external verification, all duplicate review to-dos are closed, and no new binary correctness labels are added.

Apply with `scripts/apply_answer_repair_final_confirmations.py`, bound by review material and result fingerprints, not copied across outputs by sample_id. New checks cover nulls not being filled in, preferences not being misrecorded as harmful repairs, different answers to the same question staying independent, the F37 re-judgement, missing scope and protection of the existing C1 human review; together with the related existing checks, 14 checks pass. No new GPU calls, strategy changes or scoring changes were made.
