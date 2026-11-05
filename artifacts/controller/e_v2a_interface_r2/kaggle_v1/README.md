# r2 First GPU Result · 2026-10-26

[Kaggle Version 1 / 362285592](https://www.kaggle.com/code/yearo111/controller-e-r2-shared-frozen-20261026?scriptVersionId=362285592) has finished, with a page runtime of 534.1 seconds, GPU T4×2. The actual model is on cuda:0, shared Qwen2.5-7B / NF4, revision `a09a35458c702b33eeacc393d103063234e8bc28`; the original environment file is in [environment.json](raw/environment.json).

**Conclusion: r2 did not pass engineering acceptance and stopped per preregistration. NQ, FEVER, and all groups containing E are uniformly unavailable; the mainline no longer schedules r3.** This time no real development set, selection-run set, or final evaluation was run, so no new EM, net repair gain, or real development compliance rate can be reported.

## Integrity and Actual Results

- [outputs.zip](outputs.zip) is the original download, 118,102 bytes; SHA-256: `9cf94fb3e8573213e0a86462690b7291b8801a176423e91e949d501323c2f28d`, consistent with the Kaggle log.
- [raw/](raw/) keeps the full raw-byte unpacking; [audit.json](audit.json) is independently recomputed from the ZIP, with the 26 checksums and the frozen input manifest consistent, and the source is `e3ddead`. **A passed audit means the archive is trustworthy, not that E passed.**
- All 18 short questions and 2 long questions ran: 10 passed, 1 structural failure, 9 semantic failures, 0 execution failures.
- NQ has 11 items in total, 7 passed; FEVER has 9 items in total, 3 passed; each includes its own 1 long question. This is a fictional acceptance that deliberately covers boundary behavior; the ratio is not benchmark accuracy.
- The two long inputs are 17,514 and 17,539 tokens, with XGrammar 0.2.8 and prefill256 both enabled; execution and the output contract passed, while the repair behavior failed in both.
- [gate_summary.json](gate_summary.json) records 20 model calls: 58,096 input tokens, an estimated 1,586 output tokens, and total model time of about 328.3 seconds. The two long questions took about 99.3/103.0 seconds each, and the short-question median was about 6.5 seconds; this is acceptance cost, not development cost or AC5 end-to-end cost.

The red Notebook failure comes from the preregistered exit 3, converted by the outer layer into a RuntimeError; it is not a CUDA crash, OOM, or loss of the result ZIP. Because a structural failure exists, the condition "run the 400 verify-only development calls only if there are only semantic failures" is not met; actual development calls are 0.

## What the Raw Records Show

| Issue | Reviewable record | Interpretation |
|---|---|---|
| Correct candidate paired with the wrong evidence sentence | [NQ repair-1](raw/engineering/v2a-r2-fiction-nq-contradicted_repair-1.json) | The candidate Ilsa Marrow is the correct name, and the output cites R2S2; that sentence only says Kerran Pell made the frame, while the name is in R2S1. The contiguous-span check rejected the replacement, kept the original answer, and marked it unresolved. This is the 1 structural failure; a valid sentence ID does not guarantee the candidate exists in that sentence. |
| Conflict treated as adjudicable | [NQ conflict-2](raw/engineering/v2a-r2-fiction-nq-conflict_insufficient-2.json), [FEVER conflict-1](raw/engineering/v2a-r2-fiction-fever-conflict_insufficient-1.json) | The rationale already noted that the two pieces of evidence conflict, yet it still judged contradicted and chose one side to change the answer; none of the 4 conflict questions passed. The evidence-source check cannot block this kind of semantic error. |
| FEVER judgment target confusion | [FEVER supported-2](raw/engineering/v2a-r2-fiction-fever-supported_keep-2.json), [FEVER repair-1](raw/engineering/v2a-r2-fiction-fever-contradicted_repair-1.json) | When the claim is refuted and the original label REFUTES is correct, it judged the original answer contradicted; when the claim is supported and the original label REFUTES is wrong, it judged supported and kept it. It is sometimes judging the claim rather than whether the input label is correct. |
| Missing fact still treated as refutation | [FEVER missing-2](raw/engineering/v2a-r2-fiction-fever-missing_insufficient-2.json) | There is no discoverer information, yet it changed to REFUTES. Both NQ missing-fact questions passed; this cannot be taken to mean the cross-dataset issue has been solved. |
| Rationale and action inconsistent under long input | [NQ long](raw/engineering/v2a-r2-fiction-nq-contradicted_repair-1-long.json), [FEVER long](raw/engineering/v2a-r2-fiction-fever-contradicted_repair-1-long.json) | The NQ rationale says the original answer is refuted, yet the label is supported; FEVER judged contradicted but did not provide an evidence-backed opposite label. Being able to run long inputs does not mean being able to verify correctly. |

The conclusion this supports is: constrained decoding and source checks can constrain outputs and block some invalid replacements, but still cannot provide reliable judgment of evidence semantics. It does not prove that enlarging the dataset, switching GPUs, or further changing prompts would solve the problem; nor was r2's real-data gain measured. The actual defects were observed on this round's controlled questions and cannot be extrapolated to overall model accuracy.

## Mainline Next Steps

For the deployment decision, see [../deployment_decision.json](../deployment_decision.json). At runtime, unavailable does not call the E model; it keeps the answer that entered E and records verification as incomplete; it does not fabricate supported/insufficient conclusions. An engineering-unavailable warning must not be passed off as a verified semantic-failure label to claim that AC6 is met. The FR12 requirement for the E repair path to work end to end has still not been accepted and must be disclosed in the report.

Continue T5 according to the [T1–T9 checklist](../../../../docs/CONTROLLER_TASKS.md): complete the [17-question human review](../../t5_assisted_review_v1/review/review17.html), replay the preregistered NQ routing grid with the fixed unavailable mode, freeze at most one candidate, then do one selection run. FEVER routing is not changed; P60 is not enabled, Diagnosis is not retrained, and new E revisions are not mixed into the routing candidates. Subsequent selection runs and the final six groups must record the same mode; the Always Repair group will then be a keep policy that does not execute E, and that group's zero harmful repairs must not be written up as safety evidence for a working E.

Review command (does not call GPU, does not read gold):

```bash
python scripts/audit_controller_e_v2a_outputs.py --archive artifacts/controller/e_v2a_interface_r2/kaggle_v1/outputs.zip --release-dir artifacts/controller/e_v2a_interface_r2/gpu_release_v2
```
