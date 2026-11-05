# E v8 NQ75: Kaggle results and output contract failure

On 2026-10-19, Kaggle version 2 ran successfully in 4,481.3 seconds (74.7 minutes), four groups of 75 records each, 300 real predictions in total. Inputs and model configuration are the same as the frozen experiment package. **Full v8 failed the original answer check on all 75 records and fell back, without executing the subsequent repair; this round cannot demonstrate its semantic protection or repair gain.** The default strategy is not promoted.

## Completeness and reproduction

- [Kaggle version 2 results](https://www.kaggle.com/code/yearo111/e-v8-nq75-conservative-gates-20261019/output?scriptVersionId=359843897), T4 ×2; the first loading failure and this success are recorded separately.
- [Original result ZIP](../artifacts/answer_repair/v8_nq75/answer_repair_v8_nq75_outputs_export.zip), SHA-256 `0bf8aa78707b9764ef07dab0948ad029240f969f230ba823c2365fd39772277f`, 27 files, CRC and unpacked content hashes pass.
- Original input package SHA-256 `906f096f78e36a4f272cdd863ce2e7383d62acb638c1e480a06c5d3f75daa7a3`; all 85 payload files were checked in the cloud. The original package, original outputs, reference answers and scoring were not rewritten.
- [CPU decision replay](../artifacts/answer_repair/v8_nq75/release_audit/cpu_replay.json): replays all 300 records with the frozen algorithm inside the input package; decisions, stage traces, calls and costs pass the check, with no new GPU calls.
- [Full validation record](../artifacts/answer_repair/v8_nq75/result_validation.json): the re-aggregation matches the original report's counts, decisions and review queue; 16 F1 aggregate fields have floating-point rounding differences of no more than `1.11e-16`, checked with a `1e-12` tolerance. The original and regenerated review queues are fully identical.

Passing the replay only shows that the results are traceable and the program decisions reproducible; it cannot verify that answers are semantically correct.

## Four-group results

The 75 records are NQ Train finalized samples already involved in development: 11 internal E targets, 64 correct controls, with 25 unresolved excluded; they are not a blind review, an overall random sample or an unseen test set. C1's human-correct count is 64 and its frozen EM-correct count is 20; the two criteria cannot be mixed.

| Group | Strict EM correct count (initial→final) | EM changed to correct/broken | Answer changes | Qwen calls | Total generation time |
|---|---:|---:|---:|---:|---:|
| C1 original answer | 20→20 | 0/0 | 0 | 0 | 0 |
| v7 `short_answer_quote` | 20→23 | 5/2 | 14 | 75 | 477.2 seconds |
| v8 `conservative_repair` | 20→20 | 0/0 | 0 | 75 | 1,657.3 seconds |
| Without original answer check | 20→20 | 0/0 | 1 | 77 | 459.8 seconds |
| Without candidate check | 20→20 | 0/0 | 0 | 75 | 1,644.3 seconds |

v7's mean F1 is `0.583828→0.618655`, the without-original-answer-check group is `0.583828→0.580751`, and the rest are unchanged. 302 generations were actually made in total, below the 600 budget upper bound, because most records fell back early. Time is the generation time recorded by each stage, excluding loading and all other cloud overhead; this is not an equal-call-budget comparison.

v7 changed answers on 4 of the 11 targets and on 10 of the 64 correct controls; these counts are not human changed-to-correct/broken. The semantic net gain of the new answers is still null and must be reviewed.

## Primary failure: all original answer checks invalid

Both full v8 and the without-candidate-check group got `original_check_invalid: 75`:

| Original answer check rejection reason | Records per group |
|---|---:|
| Does not follow the strict 9-line output contract | 65 |
| Quote cannot be located as a contiguous original text span | 10 |

In real generation, 46 records repeated the `ENTITY:` label; some outputs also had extra paragraphs, making the total line count exceed 9. 28 records exhausted the 384-token budget. The code rejected them by design and returned C1. Other outputs did conform to 9 lines, but their quotes added surrounding quotation marks or rewrote the original text's representation, and failed the contiguous slice check.

For example, N1's answer was originally near Essex Farm, but Qwen repeated the `ENTITY…REASON` checklist continuously and was eventually truncated. N12's output had 9 lines, but the quote failed original text matching; its reason also misread the question's music video filming location as "the song itself was not filmed", showing that semantics still need checking even if the format is solved.

So "64 correct controls were not modified" cannot be interpreted as the protection mechanism successfully identifying correct answers; they all took the abnormal fallback. The two experiment groups with the original answer check never reached the stage where they differ, so this round cannot measure the ablation effect of the candidate check.

Earlier unit tests verified program behaviour given compliant/non-compliant outputs, and did not prove that the real model would follow the format. This real run exposed that engineering gap; small-batch stage acceptance with the real model must be added before expanding runs.

## Other failures and semantic risks

v7 has 44 `invalid_output`: 19 with RANK not a pure number, 17 quotes that cannot be located, 8 quotes over 60 words; plus 1 subject-word constraint rejection. The without-original-answer-check group has 57 invalid candidates (42 quote location, 9 rank number, 4 missing RANK, 1 overlong quote, 1 non-three-line output), plus 1 invalid candidate check, 12 insufficient-evidence fallbacks and 4 candidates identical to the original answer.

In the without-original-answer-check group only N2 changed answer: `San Bernardino Mountains` became latitude/longitude, and F1 dropped, but this alone cannot determine a semantic error. It comes from a confirmed correct control, and still needs review of whether the change was necessary and whether it matches the requested location granularity.

v7's instances show that there is still a gap between program traceability and actually answering the question:

- N15: `inflation` was replaced with `rise in the price level`, replacing the requested term with the question's description, a clear harmful repair risk.
- N24: `A. Mitchell Palmer` was replaced with `the United States Department of Justice`, replacing "who led" with the executing agency, a clear relation/type error risk.
- N31 and N48 were changed to Prince and Govind Ballabh Pant Sagar respectively, matching the previously identified work-distinction and comparison-relation targets; N96 was changed to the naming reason of pulling bread apart like a monkey, but EM is still 0. These new outputs need separate confirmation.
- Some EM improvements come from adding place names or nouns to already correct answers: e.g. Seoul→Seoul, South Korea, Mohs→Mohs scale; not all 5 EM changed-to-correct can be counted as semantic repairs.
- `fairy tale`→`English fairy tale` caused an EM regression while semantics may remain correct; not all 2 EM changed-to-broken can be counted as semantic damage.

The above is assisted analysis and has not been written into the human final labels.

## Human review and subsequent order

The original [300-record output review queue](../artifacts/answer_repair/v8_nq75/kaggle_20261019/output_review_queue.jsonl) keeps pending human fields; prioritise reviewing the [15 changed-answer records](../artifacts/answer_repair/v8_nq75/review/changed_outputs_pending.jsonl), each containing the question, before and after answers, full Top-5, offline reference and the original C1 review basis. For v7 they are N12/N15/N24/N29/N31/N47/N48/N52/N54/N68/N71/N84/N96/N100, and for the without-original-answer-check group N2. The numbering follows the original NQ Train first batch of 100.

1. **First fix the output contract and generation stopping conditions.** Specify a single check output, and test the acceptable representations of rank numbers and quotes; any representation normalisation must still locate the real original text, and cannot use fuzzy matching, delete negations or substitute entities, nor pick the favourable conclusion among repeated but contradictory checks. Keep this run's strict outputs and failure counts.
2. Add the actual failing raw text to regression checks, and run a real-model smoke on a small number of fixed development examples, to confirm that the original check, repair and candidate check can actually be entered and can reject. A full run must not be submitted merely because local synthetic outputs pass.
3. Save the newly revised prompts/code/config and results separately, rerun the four groups on the same 75 records, and count format validity rate, target repairs, correct-control harmful repairs and call cost. Offline lenient-parsing replay cannot be treated as a new model experiment.
4. In parallel, have humans confirm these 15 new outputs and check unrepaired targets. The 25 old unresolved records remain isolated; there is no need to wait for all of them to be completed before fixing engineering problems.
5. Only after the pipeline works normally, judge whether same-model checking is effective; after fixing the strategy, confirm with the team a shared evaluation set not involved in development, then expand natural errors and correct controls. The current failure is first an output contract problem, and expanding data or replacing the dataset cannot solve it.

This round did not modify the algorithm, did not rerun the GPU, did not change human labels, and did not overwrite v7 historical results. Full v8 has not yet passed validity acceptance.
