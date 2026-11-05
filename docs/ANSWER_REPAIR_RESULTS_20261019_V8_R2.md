# v8r2 smoke: entry restored, candidate rejection and semantic checking still need correction

Kaggle version 1 `359869429` successfully completed 32 real outputs of 8 records ×4 groups. The result ZIP SHA-256 is `a1e1d549e5887682be0b554695fb41f412271ffec733e2e303b6fc49489ce387`, 28 files in total. The input package is `ca318d513f5feab0ff2f4c94acf19888e46ef0021ec454ef6dfd43f166bf58fe`, and 93 payload files have been checked.

The original 75 records and their results remain unchanged. This run is only for engineering acceptance and cannot estimate overall or unseen-set gains.

## Actual progress

- All 8 original answer checks of the full version are valid: 5 judged supported, 3 unsupported; the without-candidate-check group also has 8 valid. On the same subset the original version was entirely original_check_invalid, so the entry problem has been corrected in real generation.
- All three stage types were actually executed: 16 valid original_check, 5 valid minimal_repair, 3 valid candidate_check; of these, 1 record took the full three-stage path.
- Of the 33 v8 stages, 9 were ended by the output boundary stopper and 24 were not ended by that stopper (possibly EOS or the limit, which cannot be determined from this alone). Actual generations were not trimmed post hoc.
- But there were still 9 minimal_repair validation rejections, the preset engineering threshold was not passed, so it is not expanded to 75 records, and the threshold is not modified to retroactively approve this run.

## Four-group results

The initial frozen EM is 1/8; the initial human grouping is 4 correct controls and 4 E targets, and the two criteria cannot be mixed.

| Group | EM correct count | Changed answers | Calls | Main status |
|---|---:|---:|---:|---|
| v7 control | 1→2 | 5 | 8 | 3 invalid_output, 5 repaired |
| Full v8r2 | 1→1 | 0 | 12 | 5 original_supported, 2 candidate_invalid, 1 candidate_rejected |
| Without original answer check | 1→0 | 1 | 10 | 1 candidate_unchanged, 5 candidate_invalid, 1 repaired, 1 candidate_rejected |
| Without candidate check | 1→2 | 1 | 11 | 5 original_supported, 2 candidate_invalid, 1 repaired |

41 generations were actually made in total, not exceeding the 64 budget. The human final labels of the 7 changed-answer records stay pending. EM does not equal semantic truth; this round cannot claim a positive improvement on that basis.

## Two kinds of problems need to be handled separately

Candidate output rejections total 9, coming from 5 distinct "question + candidate prompt" outputs (different ablations repeat the same prompt):

1. **Format omission**: N12 output three bare-value lines, and N96's first line output only `1` instead of `RANK: 1`. The full / without-original-check / without-candidate-check groups each showed these two kinds, 6 times in total. Explicit and unambiguous candidate grammar compatibility can be designed in the new revision, but original text location must still be done item by item, and arbitrary prose cannot be accepted nor a choice made among multiple sets of responses.
2. **Source constraint rejection**: 3 more in the without-original-check group. N2's ANSWER is not within the quoted latitude/longitude span; N8's answer was not located in the selected quote; N31's QUOTE deleted content from the middle of the original text. These rejections reflect source protection and cannot be eliminated through fuzzy matching, splicing or relaxing the contiguous span requirement.

Format rejections and the model proposing non-compliant evidence should be recorded separately; this round's single candidate_invalid is not clear enough. The next version should explicitly distinguish rejections due to format, source, length and other limits, and fix a new engineering acceptance definition prospectively; this round's established false is kept.

After the entry was restored, semantic misjudgements were also exposed: N8/N31, confirmed as E targets, were let through by the original answer check; N48 extracted Govind Ballabh Pant Sagar but was then rejected by the candidate check, while the group without the candidate check accepted that answer. The without-original-check group changed the correct inflation to a rise in the price level and passed self-check. This shows that the same model's separate six-item check can still wrongly accept and wrongly reject, and structural validity cannot be treated as semantic correctness.

The next step first fixes the candidate grammar and rejection classification while keeping source protection; then independently tests the reliability of the six-item check on work identity, comparison relations and term type. Changes to semantic rules must be listed as a separate new experimental factor, rather than folded into format compatibility or a post-hoc rewrite of this run's results.

## Storage and review

- [Original outputs](../artifacts/answer_repair/v8r2_smoke8/kaggle_20261019/)
- [Frozen CPU replay](../artifacts/answer_repair/v8r2_smoke8/release_audit/cpu_replay.json)
- [Result validation](../artifacts/answer_repair/v8r2_smoke8/result_validation.json)
- [7 changed answers pending review](../artifacts/answer_repair/v8r2_smoke8/review/changed_outputs_pending.jsonl)
- [Assisted opinions on the 7 changed answers](../artifacts/answer_repair/v8r2_smoke8/review/assisted_changed_outputs.jsonl): N31/N48/N96 suggested as repairs, N12 an unnecessary expansion of a correct answer, N15 a term degraded into a restatement of the phenomenon; these correspond to 5 unique samples, and records repeated across groups cannot be treated as independent samples. Human review fields were not changed, and a confirmed net gain cannot be reported on this basis.
- [Preset threshold raw results](../artifacts/answer_repair/v8r2_smoke8/kaggle_20261019/smoke_acceptance.json)

## Basis for subsequent semantic optimisation

[6 assisted diagnosis cases](../artifacts/answer_repair/v8r2_smoke8/release_audit/semantic_failure_triage.jsonl) have been compiled, containing the full Top-5, the original review snapshot, the actual stage raw text and assisted opinions. The human review fields of the new outputs are still pending.

| Sample | Actual problem | Behaviour to verify next |
|---|---|---|
| N8 | Confused Moon River in the film with the same-named song in the question; the wrong answer was kept | Precisely match the work title, medium and performer |
| N12 | The correct location was wrongly rejected by the original check, which demanded a film/MV distinction the question does not have | Avoid adding conditions out of nothing; measure correct-control false positives |
| N31 | The reason and evidence point to Prince, yet Charlie Rich was judged supported | First extract the evidence object, then compare with the answer under review |
| N48 | After correctly negating the second-largest lake, the correct candidate was then wrongly rejected | The direction of the comparison relation and the check object must be consistent |
| N96 | The candidate lacks the field name; no actual candidate check result has been obtained yet | Fix the grammar first, regenerate, then judge semantics |
| N15 | The without-original-check group changed inflation into a restatement of the phenomenon, and the candidate check let it through | Distinguish the term the question asks for from related explanations |

v8r3 only handles candidate grammar and rejection classification, keeping the semantic prompts unchanged, to isolate the engineering impact. After that, priority is to separately test the checking method of "first extract the fact from the evidence according to the question, then compare the original answer/candidate", in a same-condition comparison with the existing simultaneous six-item judgement; this is only a route to be verified and cannot be assumed to improve. Keep correct controls and similar-work distractor evidence, and count repairs, harmful repairs, wrong answers let through, correct candidates wrongly rejected and call cost. At this point replacing the dataset is not a priority, nor is the final system configuration selected based on these 8 development samples; natural target errors and a shared test set not used for tuning still need to be filled in.

All 146 Python checks of the local isolated copy pass, and 106 portable checks in the package pass; 32 simulated pipeline records and the actual 32 GPU outputs were verified separately, and simulated records are not used as research results. Any new revision, prompt or model result is saved separately and does not overwrite this package.
