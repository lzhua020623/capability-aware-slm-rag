# T5 user-adopted assisted review · 2026-10-26

Current entry point: [confirmed 17-question web page](review/review17.html). After viewing the assisted judgments for R01–R17, the user replied "按照你判断的来"; this snapshot only implements this explicit adoption, and the original assisted opinions and the unreviewed snapshot are kept as is.

- Assisted opinions adopted: 17/17; this is a visible-reference review assisted by the assistant (agent), not an independent blind review.
- Fully finalized: 11/17. For R01, R02, R04, R05, R08, R12, R13, R14 both the original answer and the retrieval-enhancement answer are correct; for R07, R10, R15 the original answer is wrong and the retrieval-enhancement answer is correct.
- Not yet fully finalized: R03, R06, R09, R11, R16, R17. For R06 and R09 the original answer and Adaptive are judged wrong, and the retrieval-enhancement answer is still pending; for the other four questions all three correctness judgments are pending. R16 and R17 keep the operational suggestion of "leaning towards keeping the original answer", and this preference is not written as a correctness label.
- The confirmation scope covers only the semantic judgments for C1, the original Adaptive and Always Retrieve. Always Rerank and the historical Always Repair on the web page remain outside the scope of this confirmation.

This is a deliberately selected priority queue. The 3 repairs and 0 harmful repairs among the 11 finalized questions cannot represent the net gain for all 200 development questions, the 107 candidate retrieval enhancements or the routing candidate. The six pending questions are not scored as correct; all frozen gold, aliases and EM are kept as is, and are not automatically converted into training/DPO labels.

## Traceable records

- [User confirmation receipt](confirmation.json): the original words, scope, original assisted files and SHA256 of the notes.
- [Committed per-question records](review/review17.jsonl): `assistance_adopted_by_user` and `confirmed / partial / unresolved` are recorded separately.
- [Statistics](review/summary.json) and [output fingerprints](review/manifest.json).
- The [original assisted snapshot](../t5_assisted_review_v1/README.md) is still kept, and the historical "not confirmed by the user" is not rewritten as already confirmed at that time.

This round only saves the review and does not modify the models, runtime routing, r2 acceptance, the decision that E is unavailable, or the selection rules. The candidate has not yet been frozen/promoted; replay analysis must later take into account the pending items and the review coverage; accepting the assisted opinions cannot be equated with unconditional permission to send it to the selection run.

Rebuild command (the target directory must not exist):

```bash
python scripts/record_controller_t5_confirmation.py --confirmation artifacts/controller/t5_human_review_v1/confirmation.json --output-dir /tmp/controller-t5-confirmed-new
```
