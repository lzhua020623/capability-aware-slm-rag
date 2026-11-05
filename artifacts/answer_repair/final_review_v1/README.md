# E final changed-answer review list

On 2026-10-21, the final 75-item development run and the one-time 125-item challenge completed frozen replay. All changed answers of the main strategy formed 42 review items, covering 41 samples; assistant assistance 42/42; as of 2026-10-22, the user has looked through 42 items, 42 items are fully processed, 37 can be scored in binary, and 5 have been processed with scoring notes retained. The latest entry point is `F42_human_review.html`; the original `F42_review.html` is kept as the frozen display material from before human confirmation.

- F1–F14: v7 changed answers of development 75, mapped to the original N numbers.
- F15: the only changed answer of the final r3 on development 75, originally N96.
- F16–F42: v7 changed answers of challenge 125, keeping the U numbers (frozen challenge_number). The challenge r3 has no changed answers.

F13/F15 are different answers from different strategies for the same question, and are not double-counted as two independent samples. `prepare_answer_repair_final_review.py` only reuses review material whose question, initial and final answers, evidence, references and actual citations are completely identical; the source of equivalent results is retained. This batch has no completely identical material to merge. Ablation records are not in the main review scope; full predictions are still saved.

Open `F42_review.html` to view the question, C1, changed answer, assistant opinion, actual citations and full Top-5; searchable by number. Reply in chat with "F1–F5 agree" or correct items one by one. The page is read-only and does not automatically submit human labels. The H prefix continues to be used for the historical 45 items; the N prefix refers only to the original development numbers.

| Stratum / strategy | Suggested fix | Suspected harmful repair | Correctly preserved | Correctness pending |
|---|---:|---:|---:|---:|
| Development v7 | 3 | 2 | 8 | 1 |
| Development r3 | 1 | 0 | 0 | 0 |
| Challenge v7 | 2 | 4 | 15 | 6 |

The above is the assistant's draft snapshot from before human confirmation, and cannot be taken as the current human conclusion or the final gain. The key pending items in the draft were F4, F19, F21, F22, F35, F36, F40; the other 35 items also need confirmation. F32 still needs confirmation of the FIFA event and the corpus time point, and F30 needs confirmation of the degree vs diploma criterion.

`pending_pairs.jsonl` saves the frozen material, the original result SHA and all pending human fields. `assisted_partial.jsonl` keeps the historical tool naming, but actually already contains all 42 assistant opinions, with verbatim source locations. `manifest.json` freezes the union scope; `assisted_summary.json` records the subsequent assistant completion status. Unknowns stay null, and existing initial human correctness is not overwritten by the assistant.

Materials are rebuilt with `scripts/prepare_answer_repair_final_review.py`; the summary and web page are generated with the `--directory artifacts/answer_repair/final_review_v1 --prefix F --title 'E · 最终42条改答审核' --summary-version answer_repair_final_changed_assisted_complete_v1 --scope '<full batch and limitation description>'` arguments of `scripts/summarize_answer_repair_historical_review.py`. No new model calls, no strategy changes.

For the full numbers and limitations see `docs/ANSWER_REPAIR_RESULTS_20261021_FINAL.md`. Human processing of F1–F42 is complete. Human confirmation of the historical H1–H45, the end-to-end routing experiment and team data-isolation coordination are still pending.

## Current human confirmation update (cumulative batch 4, 2026-10-22)

F12/F23/F25/F36 are recorded as keeping the original answer, but the correctness of the changed answer is pending; F37, under the criterion that the aid recipient must be specifically stated as Continental Army, is recorded as both answers wrong. The other agreement opinions keep their existing null. After the additional "F4 supported", F4 is recorded as a fix; F12 again requested keeping the C1 original answer, and short-answer correctness is still pending. Batch 3 confirmed F21 original right changed wrong, F22 both answers right, F35 original wrong changed right, F40 both answers wrong. F12/F23/F25/F36 have been decided to keep the original answer, and F19 has completed external verification; these 5 items only retain scoring notes and are no longer to-dos. The old table is the assistant's draft snapshot; for current human statistics see `data/annotations/human/answer_repair_final_v1_batch04_20261022.summary.json`, the original wording and clarifications are saved separately with the same prefix `.confirmations.json`, and for the detailed explanation see `docs/ANSWER_REPAIR_FINAL_HUMAN_REVIEW_20261021.md`.

Development v7 confirmed fixes 4, harmful repairs 2, pending 1; development r3 confirmed fix 1, no harmful repairs or pending; challenge v7 confirmed fixes 3, harmful repairs 4, pending 4. Unresolved items were not imputed as wrong or correct, and no cross-strategy aggregate net gain was computed.
