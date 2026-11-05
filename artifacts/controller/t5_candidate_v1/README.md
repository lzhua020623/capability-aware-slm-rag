# T5.8 development replay and the single candidate · 2026-10-26

**NQ threshold=0.2** has been frozen from the original grid, only as the candidate for a single selection run, not promoted. The default Diagnosis, Probe, Safe Binary, models, prompts and three modules are all unchanged; E is unavailable for all groups in both datasets. The failures of the original T4 and r2 keep their original bytes.

## Pre-freeze re-check

The assisted opinions on the original 15 questions and the 2 supplementary questions have all been explicitly adopted by the user; see the [confirmation record](../t5_human_review_v1/README.md). Of these, 11 questions are fully finalized and 6 contain pending items; "reviewed and accepted the pending items" does not mean all semantic labels have been determined. What is satisfied here is the precondition that the priority queue has been handled by the user; unknown values are not used to claim that the semantic effect threshold has been passed.

Per the original registration, the development phase uses the frozen EM grid to select at most one rule; the veto by semantic net regression/pending lower bound happens in the single selection run. No new development semantic threshold is added, and the threshold is not changed, no additional search is done and no repeated tuning happens because development semantics are unknown. The wide intervals below will be disclosed, and all answers actually changed by the new routing are required to be reviewed after the selection run.

## Development results (200 per dataset)

| NQ threshold | Original routing correct | Candidate correct | EM fixed/broken | New retrievals | Changed answers | Changed answers with complete semantic review | Semantic net lower/upper bounds |
|---|---:|---:|---|---:|---:|---:|---|
| 0.2 | 78 | 81 | 5/2 | 107 | 17 | 4 | −11 to +14 |
| 0.3 | 78 | 79 | 2/1 | 64 | 11 | 1 | −9 to +10 |
| 0.4 | 78 | 78 | 0/0 | 26 | 4 | 0 | −4 to +4 |

FEVER's original routing correct count is 174/200; answers, actions and replay costs are identical under all three rules. The inputs, final answers, evidence and module call parameters of the 800 cached source Adaptive/No Probe records all match the fixed-action groups. The 16 NQ items routed to E, with E unavailable, keep the original answer that entered E and keep only the existing Probe calls; E verification is neither executed nor fabricated.

Threshold 0.2 has 3 more EM-correct answers than the original routing (1.5 percentage points), but **a semantic net gain cannot yet be confirmed**. Of its 17 changed answers, 4 have complete confirmation; 4 are still pending in the existing queue (R09, R11, R16, R17); 9 are not in the original priority review queue. The bounds are computed by taking the worst/best case for all unknowns, without filling in semantic ground truth with EM. There are zero answer changes that are exactly identical.

The 0.2 replay adds 104 LLM calls and 107 each of retrievals and reranking scorings; reusing measured call times, the added provider time is estimated at about 1,444.9 seconds. The estimate excludes deployment scheduling, loading and the shared C1, and is not a newly measured latency or an AC5 certification. The rule "do more retrieval enhancement" is costly and must be reported truthfully in the final comparison with Full; one must not cherry-pick only the gain numbers.

## Candidate rule and next steps

Only questions with `dataset=nq` and original action `no_action` can be overridden; an existing explicit supported Probe is a fixed veto; otherwise the existing first-level failure probability is used, and retrieval enhancement is applied if it reaches 0.2. When there is no Probe, no new calls are added and a missing judgment is not treated as unsupported. The runtime does not access gold or human-review correctness, and does not special-case by sample_id.

[candidate.json](candidate.json) is the frozen entry point; [analysis.json](analysis.json) stores all three attempts and the per-question changes; [replay_records.jsonl](replay_records.jsonl) is an offline projection with source fingerprints, not raw GPU predictions. Only development gold was read; selection/final samples have not yet been read or scored.

Next step: a selection run of 200 items for each of the two datasets, running only the original Adaptive, Always Retrieve, Always Rerank and Always Repair (E unavailable, no E model calls), with C1 reused. Candidate results are replayed from the fixed actions on the same inputs and the observed Probes; they cannot be replaced by unobserved Probes or a Full replay. After the selection run, the NQ frozen EM net gain must be >0; all changed answers of the candidate relative to the original routing receive semantic review, and if the semantic net value or the worst-case pending lower bound is negative, it is not promoted, the original routing is kept, and no other threshold is selected. FEVER routing stays unchanged and the outputs are checked for consistency.

Re-check (the target directory must not exist):

```bash
python scripts/prepare_controller_t5_candidate.py --output-dir /tmp/controller-t5-candidate-new
```
