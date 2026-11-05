# T4 development set complete results · 2026-10-25

GPU inference and frozen automatic scoring are complete. **Adaptive ties on NQ EM and has a net gain of 4 questions on FEVER; E and full enhancement still have negative net gains.** NQ semantic review, AC6 and trace human acceptance are not yet complete, and neither the T5 selection run nor the final test has been carried out.

## Are the results usable

[Kaggle V2](https://www.kaggle.com/code/yearo111/controller-t3-cuda-engineering-frozen-20261024?scriptVersionId=361785012) ran for 31,265.1 seconds (about 8.68 hours), and the page shows failure. After all 2,800 predictions were actually saved, the final summary projection omitted `total_seconds`, causing a `KeyError`. This is a summary engineering error; the raw records contain complete timing fields, and all fixed IDs, the seven-group pairing, input/prompt hashes, budgets, the runtime environment and the engineering receipts passed acceptance, with call failures and budget blocks at 0. No answers were regenerated.

The failed status in the original [manifest](raw_metadata/manifest.json) is retained, without rewriting history. The [acceptance](release_audit.json) and the [summary rebuilt from the original predictions](reconstructed_summary.json) are stored separately. The current runner only fills in the fields required by the summary; a new regression test runs from 400 fictional questions, seven groups and 2,800 saved records all the way to a successful summary. 774 full regression tests passed (85.605 seconds), and after the test streams were closed and cleaned up this regression passed again (16.795 seconds); see [validation](validation/summary_fix_tests.json). The frozen GPU package and the original predictions were not modified.

The raw ZIP is `controller_t4_development_v1_outputs.zip`, 14,232,583 bytes, SHA256: `be3b926b8f808cd39b1c32b64564e5e5619518bebd2585af3fd8f0a6616bc7a5`. Within it, the raw predictions are 127,988,603 bytes, SHA256: `972d5571b8f2e2263581c17a089a01f5c76498f7057e69e276d5206ddb30a1ad`. Git keeps the raw ZIP and small metadata; the 128MB extracted predictions are only in the ignored directory `data/processed/controller_results/t4_v1/` and can be losslessly restored from the ZIP.

## Seven-group comparison

200 development items per dataset. NQ uses frozen EM and FEVER uses frozen label accuracy; each column is number correct/200. Semantically equivalent NQ answers are reviewed separately, without modifying this round's gold or aliases. In this round E is NQ `short_answer_quote` and FEVER `verify_repair`, sharing Qwen/NF4; the conclusions apply to these two frozen policies and this batch of data.

| Comparison group | NQ EM | FEVER accuracy | NQ fixed / broken | FEVER fixed / broken |
|---|---:|---:|---:|---:|
| C1 cached baseline | 78 (39.0%) | 170 (85.0%) | 0 / 0 | 0 / 0 |
| Always Retrieve | 85 (42.5%) | 171 (85.5%) | 11 / 4 | 5 / 4 |
| Always Rerank | 81 (40.5%) | 172 (86.0%) | 5 / 2 | 5 / 3 |
| Always Repair (E) | 68 (34.0%) | 166 (83.0%) | 4 / 14 | 2 / 6 |
| Full serial full enhancement | 72 (36.0%) | 166 (83.0%) | 10 / 16 | 5 / 9 |
| Adaptive | 78 (39.0%) | 174 (87.0%) | 1 / 1 | 4 / 0 |
| Adaptive No Probe | 79 (39.5%) | 174 (87.0%) | 2 / 1 | 4 / 0 |

NQ token F1: C1 is 0.553569, Adaptive is 0.553021, No Probe is 0.557711, Always Retrieve is 0.568517; for the other complete metrics see [evaluation.json](evaluation.json). Being sent into a module, generating a new candidate, or an `answer_replaced` status is not the same as broken; in the table above, broken strictly means the original answer's metric was correct and the final metric is wrong.

## Where the losses come from

**NQ routing has missed repairs.** The offline-observed per-question optimum of "keep + the three single actions after C1" is 93/200, while Adaptive is 78/200. The gap consists of 1 avoidable harmful repair and 14 missed repair opportunities; of the 14 questions, 10 kept the original answer, 2 were sent to reranking, 1 was sent to E and 1 was sent to retrieval enhancement. 10 missed repairs could be fixed by the fixed retrieval-enhancement group (the success sets of different actions overlap). This is an observed ceiling for the current action set and cannot be treated as deployable performance, a pure Diagnosis causal loss or a model-intrinsic ceiling; NQ is still affected by the EM criterion.

**E has both output-contract problems and harmful repairs after accepting candidates.** Standalone E on NQ had 111/200 invalid_output (55.5%): the quote was not a verbatim slice of the evidence 50 times, RANK was invalid 39 times, the quote exceeded 60 words 20 times, and line format/short-answer length 1 time each; in addition, 3 candidates could not be anchored. The 35 actual changed answers fixed 4 questions and broke 14 by EM. Standalone E on FEVER had 56 invalid_output, of which 46 were quotes that were not verbatim fragments, 8 were repairs that did not change the original label, and 2 were JSON parse failures; the actual changed answers fixed 2 questions and broke 6. A format fallback is not the same as a wrong final answer; all fallbacks are kept and counted in the comparison as usual; see [error classification](diagnostics.json).

The per-stage correct counts in Full are: NQ **78→85→84→72**, FEVER **170→171→171→166**, corresponding in order to C1→retrieval enhancement→reranking→E. On top of the existing preceding state, the final E stage fixed 4 and broke 16 on NQ and fixed 0 and broke 5 on FEVER; these cannot be mixed with the numbers for E executed alone from the original C1.

**The overall negative result of standalone E cannot be used to claim that it drags down the current Adaptive.** Adaptive's NQ E subset of 16 questions ended with no changed answers (13 invalid_output, 2 keep, 1 could not be anchored), net 0; Adaptive's only NQ harmful repair came from retrieval enhancement, and its only fix came from reranking. FEVER Adaptive only chose keep or retrieval enhancement, with 0 E and reranking triggers; FEVER's 4-question gain came from selective retrieval enhancement. The FEVER observed optimum is 179/200 and Adaptive is 174/200, still missing 5 repair opportunities fixable by a single action.

**Probe gave no additional accuracy gain in this batch.** NQ called Probe 56 times, and Adaptive had 1 fewer correct question than No Probe; FEVER called Probe 53 times, and the final 200 labels are exactly identical to No Probe. Additional Qwen calls were 102 vs 65 for NQ and 119 vs 67 for FEVER. This comparison only supports a development observation for this batch's configuration and cannot be generalized to all Probes being ineffective.

## Cost and statistical boundaries

| Dataset | Adaptive incremental Qwen calls / Full | Adaptive incremental token estimate / Full | Incremental tokens saved |
|---|---:|---:|---:|
| NQ | 102 / 584 | 111,759 / 488,569 | 77.1% |
| FEVER | 119 / 590 | 171,624 / 946,069 | 81.9% |

This includes Probe; the token criterion is input tokens plus re-tokenized estimated output tokens, **counting only the increment after cached C1**. The shared C1 generation/initial retrieval cost is incomplete, so this cannot be used to declare that the proposal's full-pipeline AC5 passed. Cumulative within-group time (excluding the shared C1 and resource loading): NQ Adaptive 868.4 seconds, Full 5003.9 seconds; FEVER Adaptive 1791.5 seconds, Full 8172.8 seconds. These are within-group timings accumulated during this interleaved execution, not wall-clock experiments of separate independent deployments.

This batch is a descriptive development result. Paired exact McNemar of Adaptive vs C1: NQ p=1, FEVER p=0.125; after four-test Holm, 1 and 0.375 respectively. The conservative non-inferiority test of Adaptive vs Full after Holm gives NQ 0.4262 and FEVER 0.0840; neither establishes the preregistered non-inferiority. A single bootstrap interval or a higher point estimate cannot replace the test conclusion, and "not established" cannot be written as "proven worse". The statistical methods and all intervals are retained in [evaluation.json](evaluation.json).

## Human review and follow-up

- [NQ changed-answer queue](nq_changed_answer_review.jsonl): 104 distinct "original answer→candidate" pairs, covering 84 questions; source groups are merged by identical answer pair, keeping the initial Top5 and each group's final evidence. The first 14 entries come from Adaptive changed answers. All human fields are pending, and EM scores are only auxiliary.
- [Routing priority queue](routing_priority_review.jsonl): 15 NQ and 5 FEVER observed harmful-repair/missed-repair cases. Confirm NQ semantics first, to avoid mistaking EM differences for routing failures.
- [Adaptive warnings and complete trace queue](adaptive_flagged_trace_review.jsonl): 51 entries (NQ 29, FEVER 22), keeping the complete run records. AC6 requires at least 50 human confirmations with observed precision ≥90%; the independence/completion fields for the qualitative review of at least 50 traces are recorded separately. Neither has passed acceptance, and the warnings themselves are not ground truth.

Per the preregistered T5 rules, in the current NQ losses the 14 missed repair opportunities exceed the 1 avoidable harmful repair, so NQ routing missed repairs are checked first; E contract rejections and the No Probe results are used for explanation and candidate design. No new candidate has been selected or promoted yet; this scoring/analysis did not read selection/final gold, and did not change thresholds, prompts, models, the knowledge base, training or RSI. After the semantic review is closed, only one focused candidate is made, confirmed on an independent selection set, and then the final test is frozen. There is no evidence supporting directly promoting E now to an already-effective default policy.

## Reproduction

From the repository root, in the existing complete dependency environment, run:

```bash
python artifacts/controller/t4_v1/reproduce.py --output-dir /tmp/controller-t4-offline-replay
```

The output directory must not exist, to prevent overwriting human fields. The script only reads this ZIP, the sealed engineering receipts and the development gold, and after verifying hashes/pairing/budgets it rebuilds the summary, the frozen evaluation, the loss analysis and the review queues; no model loading or GPU calls. The JSON results of the original CLI scoring and the independent ZIP replay have been checked object by object to be identical; see the [replay receipt](reproduction_receipt.json). The original inference source is 42c7974; the subsequent summary patch does not replace the archived source, the original manifest or the predictions.
