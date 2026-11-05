# T5 routing assisted review · 2026-10-26

**Follow-up status**: the user replied "按照你判断的来", adopting the assisted opinions on this page; the current committed status is 11 questions fully finalized and 6 questions not fully finalized; see the [confirmation snapshot](../t5_human_review_v1/README.md). Below, the state at the time this assisted snapshot was generated is retained.

Entry point: [17-question web page](review/review17.html). R01–R15 keep the original priority queue numbering; R16 and R17 are two new EM harmful repairs of threshold 0.2 relative to the original routing, which the original 15 questions did not cover. You can confirm by number in chat, or fill in and export on the page; filling in the browser does not automatically write back to the repository.

All judgments come from the assistant (agent); repository human-review confirmations are **0/17**. The page shows method names and references, so this is a visible-reference review with assisted opinions, not an independent blind review; the report must disclose the influence of the assistance. Human selections are not automatically pre-filled. r2 GPU has not yet run; this page shows T4's historical fixed actions, not new r2 predictions.

## Preliminary findings

- R01, R02, R04, R05, R08, R12, R13, R14: leaning towards the original answer already being correct, mainly differences in how names, regions or details are written; an EM improvement does not automatically equal a semantic repair.
- R07, R10, R15: the retrieval-enhancement group gave answers that are more on-topic and directly supported, leaning towards these being valid repairs; still pending user confirmation.
- R03, R06, R09, R11: involve roles, the relation being asked about, year/event scope or broadcast version; kept pending item by item, not forcibly adjudicated by the frozen reference.
- R16: 65 and 62 episodes both appear in the evidence, so the counting scope is unclear; R17: "current" statements for both Federer and Nadal exist, lacking a point in time. Both are harmful repairs under frozen EM; semantically a conflict is recorded first, leaning towards keeping the original answer.

These opinions cannot serve as new performance conclusions or as semantic accuracy for all 200 questions. The candidate is not frozen and not promoted; gold, aliases, routing and models are unchanged; and the assisted opinions on this page are not used directly as training/DPO labels either.

## Sources and re-check

The raw T4 ZIP, the original 15 questions and the assisted notes all have SHA256; citations are also bound to passage IDs, source-text offsets and full-text fingerprints; see [manifest](review/review_manifest.json). The existence of a source only proves traceability, not that the assisted semantic judgment is correct.

- [Assisted notes source file](source_notes.json)
- [All questions, evidence and per-question suggestions](review/review17.jsonl)
- [Original 15-question review page](../e_v2a_v1/t5_development_review/routing15_review.html)

Generation command (the target directory must not exist, to prevent overwriting review work):

```bash
python scripts/prepare_controller_t5_assisted_review.py --output-dir /tmp/controller-t5-assisted-new
```
