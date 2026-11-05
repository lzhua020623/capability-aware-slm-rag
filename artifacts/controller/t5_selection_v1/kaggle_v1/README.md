# T5 selection results · 2026-10-26

The first private [Kaggle Version 1 / 362316572](https://www.kaggle.com/code/yearo111/controller-t5-selection-frozen-20261026?scriptVersionId=362316572) completed successfully. The page records a run of 13,888.8 seconds, about 3 hours 51 minutes; 200 questions each for NQ and FEVER, 2,000 unique records across five groups, with call failures and budget overruns at 0. No model was trained in this round, and this is not the final test.

**The single NQ 0.2 routing candidate is not promoted; the original routing is kept.** Relative to the original Adaptive in the same E-unavailable mode, NQ frozen EM is 93→90/200, with 3 fixed and 6 broken, net −3 (−1.5 percentage points); the preregistered positive net-gain condition was not met. 0.3/0.4 will not be tried instead, and this EM selection decision will not be rewritten because of the subsequent semantic review.

## Raw outputs and verification

- [outputs.zip](outputs.zip): the original downloaded bytes, 7,339,856 bytes, SHA256 `2139d926fb32c6071b95ceae3035ac95c06e7da10118b9c8976e302bcb62a7bd`. 18 ZIP members; all 17 file checksums match. The original T4, E r2 and the run package were not overwritten.
- Source commit `7c281d0193a4d1e205cfbc23dca31d6d4a0ac861`, input package SHA256 `151676a8c07008257086701c5dbced4fdc0b5e0f98f75f6e88caf98dcbae8f6f`. The input/execution seals, the IDs and valid-input fingerprints of all 400 questions, and the offline reference fingerprints all match one by one.
- The six active long-input paths passed, with actual inputs of at least 16,005 tokens; the two zero-call checks with E disabled passed. The ordered Top-5 reproduction of the two historical queries passed. The cloud preflight used the original sealed package, which contains no QA gold.
- The entry state, final answers, final evidence and non-Probe call parameters of the 400 original Adaptive records are exactly identical to the corresponding fixed-action results. The candidate reuses these results and the original Probes, with no new GPU calls.
- The original summary was independently recomputed from all predictions and matches. The `answer_replaced` status count cannot be treated as the number of actual changed answers, let alone as the number of successful repairs.

## Frozen scoring

Each cell is number correct/200. NQ is normalized EM and FEVER is binary label accuracy; the two are reported separately.

| Comparison group | NQ | NQ fixed/broken vs C1 | FEVER | FEVER fixed/broken vs C1 |
|---|---:|---:|---:|---:|
| C1 cached | 89 (44.5%) | 0/0 | 163 (81.5%) | 0/0 |
| Always Retrieve | 89 (44.5%) | 9/9 | 167 (83.5%) | 7/3 |
| Always Rerank | 92 (46.0%) | 7/4 | 164 (82.0%) | 6/5 |
| Always Repair, E disabled | 89 (44.5%) | 0/0 | 163 (81.5%) | 0/0 |
| Original Adaptive | 93 (46.5%) | 4/0 | 164 (82.0%) | 1/0 |
| New NQ 0.2 candidate, offline replay | 90 (45.0%) | 7/6 | 164 (82.0%) | 1/0 |

The observed net gain of the original Adaptive relative to C1 is NQ +2.0 and FEVER +0.5 percentage points; the exact two-sided McNemar p is 0.125 and 1.0 respectively. This is a description of the selection set and cannot claim that a significant improvement has been established. The 95% paired bootstrap interval of the candidate's difference relative to the original Adaptive is [−4.5, +1.5] percentage points, p=0.5078125; the reason for rejection is not reaching the registered selection threshold, not proof of a statistically significant deterioration. Full was not run, the final four-test Holm family is not applied, and AC5 is not certified.

## Routing and cost

The candidate additionally sends 103 NQ questions to retrieval enhancement, actually changing 15 answers. Compared with the original Adaptive, the replay estimates 100 more Qwen calls, 103 retrievals and 103 scorings, with total tool/model call time increased by about 1,347 seconds. The original Probes are kept; this time is a replay estimate from existing calls, not the deployment latency of a new system. FEVER per-question answers, actions and replay costs are completely unchanged.

The measured incremental calls of the original Adaptive are NQ: 79 Qwen, 9 retrievals, 27 scorings; FEVER: 111 Qwen, 63 retrievals, 63 scorings. Detailed measured values for each group are in [audit.json](audit.json). All costs are after cached C1 and exclude the historical generation cost of the shared C1.

## E and semantic review boundaries

E is uniformly unavailable for both datasets; Always Repair keeps the input answer for all 400 records, with zero E model calls; the original Adaptive's 10 NQ E routes also fall back to keeping, and FEVER E routes are 0. This result cannot prove that E repair is effective, that verification is accurate, or that AC6 is met.

[candidate_review15.jsonl](candidate_review15.jsonl) stores all 15 changed-answer pairs of the candidate relative to the original Adaptive, the questions, the original Top-5, the candidate evidence and the frozen references; all human semantic fields are pending. The 6 NQ EM regressions do not mean that 6 real semantic harmful repairs have been confirmed; for example, an expanded person's name versus a short reference name can produce an EM disagreement. The review is used to explain the results and for the report, and the selection questions are not turned into training or tuning for the next round. Other reviews for T4 and T9 are still kept separately.

## Reproduction and follow-up

[audit.json](audit.json) is the complete automatic verification and scoring, [decision.json](decision.json) is the rejection decision, and [candidate_replay.jsonl](candidate_replay.jsonl) is the 400 offline counterfactual results; none of them are new GPU predictions. The 8 new regression tests are in `tests/test_controller_t5_output_audit.py`; all 965 full tests passed (139.540 seconds), and the five result files generated from the raw ZIP in another process are byte-for-byte identical. The log of the first local full check, which failed due to missing dependencies, is kept; after the dependencies were added, the re-check of the 3 failure paths and the full regression both passed; GPU was not rerun. Validation records are in [self_check.json](self_check.json).

```bash
python scripts/audit_controller_t5_selection.py \
  --archive artifacts/controller/t5_selection_v1/kaggle_v1/outputs.zip \
  --output-dir /tmp/controller_t5_selection_independent_replay
```

This command requires the target not to exist, to protect the historical audit. Next step: keep the original routing, close the review, register the final evaluation scope and enter T7 sealing; T8 has not started. Distillation, RSI and E semantic revisions must each be set up as a separate extension with data isolation, and this selection feedback must not continue to be used to search for new candidates.
