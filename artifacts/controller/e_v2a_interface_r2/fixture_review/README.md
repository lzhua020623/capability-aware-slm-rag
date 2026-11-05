# E r2 Fictional Acceptance Questions: Independent Blind Review and Sealing (T5.1)

Written and sealed on 2026-10-26, before any r2 prompt was modified. The questions and expectations are in [e_v2a_r2_fixtures.json](../../../../configs/controller/e_v2a_r2_fixtures.json); fingerprints and review records are in [e_v2a_r2_fixture_seal.json](../../../../configs/controller/e_v2a_r2_fixture_seal.json) (SHA256 `423aef2a…`). At sealing time, `src/compensation/evidence_v2a.py` was still the original r1 text.

## Question Composition

All 18 short questions use new fictional entities and do not reuse Selen, Veyra, Velora, or Nerin.

| Dataset | Category | Questions per category |
|---|---|---|
| NQ | supported keep, repairable error, error without replacement, missing fact, evidence conflict | 2 |
| FEVER | supported keep, repairable error, missing fact, evidence conflict | 2 |

One of FEVER's supported-keep questions has the original label REFUTES; it is used to check that "supported" means the original label is correct, not that the claim holds.

The two long questions use the first repairable-error question of NQ and of FEVER respectively as their base, padded at the end of paragraph 5, with lengths falling between 14,000–32,384 tokens. Estimated with the local Qwen tokenizer and the r1 prompt, about 17.4k; at runtime it is recomputed with the r2 prompt.

Judgment compares only four items: verification category, final action, whether unresolved, and final answer. The two NQ repair questions have preregistered harmless wording variants ("Ilsa Marrow alone", "in 1887"). Key evidence sentences are only recorded and are not a pass condition.

## Blind Review

Each round is done by a new independent assistant. It sees only the definitions and the shuffled, anonymously numbered questions; it cannot see the expectations, question categories, prompts, or repository.

| Round | Scope | Consistent with expectation | Minor ambiguity | Severe ambiguity | Subsequent changes |
|---|---|---:|---:|---:|---|
| 1 | 18 questions | 18/18 | 3 | 0 | Conflict sources changed to equal standing; "turned back" changed to "never reached the summit"; removed a personal name close to the real city Ostrava; the definitions added a conflict-priority rule and a shortest-span requirement; reduced wording leakage (attribution added to non-conflict questions, attribution removed from one conflict question, a near-miss distractor date added to one missing question) |
| 2 | 18 questions (after changes) | 18/18 | 7 | 0 | Removed a club with the same name as the mountain range; the two conflicting evidence passages changed to symmetric wording, and added "must not invent events such as recasting or rebuilding to reconcile the conflict"; clarified "when judged wrong but with no replacement, keep the original answer and mark unresolved"; rewrote the template of one no-replacement question |
| 3 | The 3 questions modified after round 2 | 3/3 | 3 | 0 | None |

Every finally sealed question was reviewed as-is by the last blind-review round covering it; this is checked by `tests/test_controller_e_v2a_r2_fixtures.py`.

## Usage Limitations

Assistant blind review does not equal human ground truth; it is used only to rule out wording ambiguity.

After sealing, the questions, expectations, and acceptable answers are no longer changed. After GPU results come out, prompts must not be adjusted around these questions and rerun. Passing these fictional questions only shows that the label contract can work on controlled inputs; it does not indicate benchmark accuracy, correct entailment judgment, or gains.
