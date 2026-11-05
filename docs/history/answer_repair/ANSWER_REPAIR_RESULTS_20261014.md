# E development experiment results: 2026-10-14

## v5 completed: same-model review did not improve results

Colab T4, 28 fixed training development samples, four groups totalling 112 real outputs and 123 Qwen calls. GPU package SHA-256 `4f115e81d9698e60f3db6f29d2b018c0388f8daf6f3ca7ed4a3b74723485b9dd`; output ZIP `2a43425501468e666087fa26df4627ad648847ff673ff49156c8e3addc8e93d7`. All outputs have been archived and passed CPU replay; the original candidates are identical in the query/guarded groups. Replay verifies parsing/decisions/cost and does not re-verify the model's factual judgements.

| Dataset / strategy | Frozen correct (14 samples) | Fixed / harmful repair | Changed answers | Calls | Generation seconds |
|---|---:|---:|---:|---:|---:|
| FEVER / v4 grounded | 4 → 6 | 2 / 0 | 2 | 14 | 44.80 |
| FEVER / v5 query | 4 → 4 | 0 / 0 | 0 | 14 | 82.53 |
| FEVER / v5 guarded | 4 → 4 | 0 / 0 | 0 | 20 | 123.36 |
| FEVER / reanswer | 4 → 4 | 0 / 0 | 0 | 14 | 84.24 |
| NQ / v4 grounded | 2 → 4 | 2 / 0 | 4 | 14 | 73.18 |
| NQ / v5 query | 2 → 2 | 0 / 0 | 3 | 14 | 98.37 |
| NQ / v5 guarded | 2 → 2 | 0 / 0 | 3 | 19 | 117.74 |
| NQ / reanswer | 2 → 2 | 0 / 0 | 2 | 14 | 18.26 |

These are raw FEVER accuracy / NQ EM. NQ has 5 human-confirmed correct controls, but the old EM recognizes only 2. Both v4 NQ EM repairs occurred on the already-confirmed-correct Columbia and bias controls, so they cannot be interpreted as real error repairs. Paired human correctness/evidence support for all new outputs has not yet been accepted; semantic metrics are null.

FEVER same-model review accepted 6 samples, 3 of which are still opposite to the reference label (Eva, Daggering, Bea). Six all-yes answers provide no independent support, and the six extra calls did not repair errors. In the NQ query group 8/14 outputs are invalid, mainly because the answer is not in a contiguous quote or the quote's punctuation was rewritten; some answers may be semantically improved, but there are also wrong events and partial answers, so not all checks can simply be relaxed.

Raw files: `artifacts/answer_repair/v5_reviewed/colab_20261014/`; validation report `artifacts/answer_repair/v5_reviewed/cpu_replay.json`. In the two-stage experiment, the old summary field `*_observed_calls` actually counted the number of samples with token records; `extra_calls` and token totals are correct. From v6 onward observed_samples is added and observed_calls is fixed, without rewriting the frozen v5 results.

## v6r1 completed: independent NLI gives local repairs, verbatim extraction regresses EM

See `ANSWER_REPAIR_V6.md` for the protocol. The latest version v6r1 has been implemented and completed 81 CPU tests, the same 81 tests inside the isolated archive, full pre-checks for all four groups, and hash verification of 48 files. Using the real fixed tokenizer on 14+50 FEVER inputs produced 365 windows in total, at most 9 windows per sample; all non-whitespace evidence characters are fully covered and no input exceeds 512 tokens. This check involved no model forward pass and cannot count as an NLI effectiveness experiment.

The latest E adapter also replayed the existing 100 real ControllerDecisions (using the frozen upstream schema classes): 71 no_action, 26 retrieval_enhancement, 2 evidence_refinement, 1 generation_repair; historical E behaviour and single-call behaviour are fully preserved. This is a CPU compatibility replay, not a new GPU effectiveness result.

The Kaggle T4 x2 environment completed real inference for all four groups: 128 outputs in total, 123 unique dataset/sample_id. The reviewed cohort overlaps the old dev50 by 1 FEVER and 4 NQ samples; they are reported separately below and cannot be combined as independent samples. The GPU session was released after the archive was downloaded and verified.

| Dataset / development batch | Frozen correct | Metric fixed / metric harmful repair | Changed answers | invalid / conflict / uncertain |
|---|---:|---:|---:|---:|
| FEVER / reviewed 14 | 4 → 7 | 3 / 0 | 3 | 0 / 4 / 3 |
| FEVER / original dev50 | 45 → 44 | 1 / 2 | 3 | 0 / 5 / 7 |
| NQ / reviewed 14 | 2 → 0 | 0 / 2 | 10 | 4 / 0 / 0 |
| NQ / original dev50 | 20 → 7 | 0 / 13 | 26 | 24 / 0 / 0 |

FEVER uses accuracy, NQ uses frozen EM. F1 on the 14 reviewed NQ samples is 0.2920→0.3164, and on the 9 core error samples 0.0202→0.2399; but F1 on the 5 correct controls is 0.7813→0.4541. F1 on the old dev50 is 0.5489→0.3756. EM regression must be separated from semantic harmful repair: many verbatim extractions return whole sentences, which get EM=0 even when they contain the correct answer.

| Dataset / batch | Qwen calls / generation seconds / input tokens | NLI forward batches / classification seconds / input tokens |
|---|---|---|
| FEVER / reviewed 14 | 0 / 0 / no generation | 29 / 12.75 / 21,910 |
| FEVER / original dev50 | 0 / 0 / no generation | 101 / 31.50 / 62,352 |
| NQ / reviewed 14 | 14 / 73.59 / 11,797 | 0 / 0 / 0 |
| NQ / original dev50 | 50 / 332.66 / 46,385 | 0 / 0 / 0 |

Cost fields follow the pre-run v6 protocol; model download/initialization is not included in the per-sample times above. The two model types have different budgets, so NLI cannot be regarded as an equal-budget replacement for Qwen. v5 ran on Colab and v6r1 on Kaggle with different dependency environments, so cross-version latency is not a controlled speed comparison.

### Detailed result notes

Among the 10 reviewed FEVER core samples, Daggering `156493`, Sugarland Express `203615`, and Greek language `72578` changed to the reference label; none of the 4 correct controls changed answer. Sugarland Express also appears in dev50 and cannot be double-counted as an independent repair. The candidate citations directly concern dancing, Spielberg's co-writing, and the use of Greek in Albania respectively, but human assessment of support for the new outputs is still pending.

The two FEVER metric harmful repairs in dev50 expose an entity problem: `140820` asks about Men in Black II, but NLI cited the 2015 Belgian film Black and changed to REFUTES with contradiction 0.9792; `33982` asks about Henry Cavill's Superman film, cited Henry's Crime, and changed to REFUTES with contradiction 0.9253. Neither the high threshold nor quote-source validation prevented an unrelated entity from triggering a verdict. Therefore the strategy cannot be promoted based on the improvement on the 14 reviewed samples.

The NQ extraction mechanism successfully preserves verbatim characters, but merges answer and evidence into one long span. `5259818422556035007` changed from Ringo Starr to a whole sentence containing that name; `970104939222709777` changed from Abbot Suger to a whole art-history sentence; both went from EM=1 to 0, which is not yet enough to call them semantically broken. Conversely, core candidates such as the Annie singer, the Loihi definition, and the Astros opponent contain more answer information but are still EM=0 and need human judgement.

There are also cases where the source is valid but the question is not answered: `970985133701043298` asks for the category of a legislative act, and extracts "passed by a majority of both houses of Congress" without the category; `-1209334507323554133` asks for the low-setting temperature of a crock pot, and extracts instructions on the minimum liquid level and heating position, without a temperature. These are the assistant's assisted analysis and are not recorded as human ground truth. An invalid fallback keeps the original answer, which also does not mean the original answer is correct or evidence-supported.

### Archive, replay, and human review queue

Input package SHA-256: `74d40b01c96245e1caf2ac8af786bf3b05394971863bddc81a75a2a2afa45ef5`.
Results ZIP SHA-256: `3e9cd5eec2a658cf9ff5146c2b7322c9166d35367b9af34696439847589df684`.
The original 30 files are in `artifacts/answer_repair/v6r1_reviewed/kaggle_20261014/`, including predictions, raw model outputs, full-window NLI probabilities, environment, install/test/run logs, and the four group summaries. Python 3.13.15, torch 2.11.0+cu128, transformers 5.16.1, accelerate 1.14.0, bitsandbytes 0.50.2; model revisions are all pinned per the frozen protocol.

Kaggle auto-extraction loading verified all 48 files after checking the fixed checksum index and saved `input_integrity.json`; it does not claim to re-verify bytes of an original archive that does not exist. Four scenarios have been accepted: loading the adapted original archive, pre-extracted files, tampered files, and a tampered manifest. 81 tests pass in the cloud.

`artifacts/answer_repair/v6r1_reviewed/release_audit/cpu_replay.json` uses this run's own frozen code to verify all 128 records: raw output/full NLI trace replay, input and settings fingerprints, decisions, citations, cost, and summaries are consistent. CPU replay does not verify factual truth.

The full review worksheet is `release_audit/pending_semantic_pairs.jsonl`; the editable sorted copy is `release_audit/review_priority.jsonl`, and the readable version is `release_audit/review_priority.md`. All 128 human fields are pending/null; the first 42 are changed results: 17 frozen-metric regressions, 4 improvements, 21 other changes; cross-cohort duplicate IDs remain among them and are kept as two batches. First review the two FEVER suspected entity harmful repairs, then the NQ metric regressions, then the improvements and other changes. For the same initial/final answer, judge correctness and support separately; they must not be auto-filled with reference matching. The other 112 v5 records and the initial 9 boundary samples remain deferred.

The pre-agreed engineering, GPU, automatic acceptance, archiving, and handoff work for this round is complete. The research conclusion is that the current strategy has not yet demonstrated a reliable net gain: keep the negative results, do not promote it to the default strategy, and do not keep tuning thresholds on the same small development set. After human re-review, decide whether to develop "separation of short answer and independent quote, entity relevance, and fallback for not answering the question"; a new strategy must be frozen as a separate version and the team must arrange a held-out evaluation that is jointly unused for strategy selection.

## Assisted review of 128 records by the assistant (pending user confirmation)

At the user's request, assisted review has been completed record by record. The review used the frozen references and C1 evidence and is a non-blind development analysis by the assistant; all human fields remain pending/null, and the numbers below are not a human gold standard or formal semantic metrics. The 128 experiment records correspond to 123 unique samples; the review materials of the 5 duplicates were checked to be fully identical, so the judgements are reused while result fingerprints are kept separately.

| Batch | Suggested semantic repair | Suggested semantic harmful repair | Correctness boundary unresolved |
|---|---:|---:|---:|
| FEVER reviewed 14 | 3 | 0 | 0 |
| FEVER dev50 | 1 | 2 | 0 |
| NQ reviewed 14 | 6 | 0 | 0 |
| NQ dev50 | 3 | 1 | 4 |

After deduplication, 9 repairs and 3 harmful repairs are tentatively judged; these are the assistant's suggestions, among which the tibia/fibula location answer is a medium-confidence boundary case that needs focused confirmation. 7 unique samples have at least one pending correctness or support item and are not forced into a binary; there are also reference mismatches and scope issues with tentative opinions that still need re-review.

The 15 NQ frozen-EM regressions (2 reviewed, 13 dev50) are all tentatively judged as answers that are still correct, expanded into evidence-supported long sentences. The 6 unique semantic repair candidates are: tibia/fibula location, the Annie singer, the Loihi definition and location, the colon serrated adenoma definition, the Astros' most recent opponent, and the episode of Jenny's return. Frozen EM recognized none of these improvements. The real harmful repair candidate in dev50, `970985133701043298`, is exactly the one EM missed: the original statute is semantically correct, while the final "passed by a majority of both houses of Congress" does not answer the legal category.

Among 63 unique FEVER samples, the actual citations of 15 concern the wrong entity, and 1 more cites an irrelevant window within a related page; not all of them changed the answer. Support must be checked against the actual quote, not just the document title or the correct label. For example, in the DID case the actually cited window discusses only amnesia/fugue/conversion disorder and does not include the DID definition earlier on that page. Among all 123 unique samples, the correctness judgement of 19 still tentatively relies on the frozen reference label/answer, with the C1 gap explicitly stated as the reason.

Key reference mismatches include: the slow cooker low vs. warming setting confusion, the Little League question citing the Major League distance, and the Soviet Union start/end years question whose reference is in Eurasia; the pharmacy student white coat question follows the previously human-confirmed criterion of "most US pharmacy students, at the end of the first year". The paleomagnetism question keeps the user's correction rationale about rock-layer records, past geomagnetic/plate information, and time scale.

Record-by-record reading: `artifacts/answer_repair/v6r1_reviewed/assisted_review_20261014/assisted_review.md`.
Machine-readable records: `assisted_pairs.jsonl` in the same directory; each record has four assistant suggestions, an independent citation judgement, rationale, confidence, risk flags, and verbatim-located evidence snippets; `decisions.json` stores the explicit assistant judgements, and `summary.json` stores per-batch and deduplicated statistics. The generation script only validates and formats these judgements and does not automatically generate semantic labels. Subsequent human confirmation should cite the 1–128 numbering of that readable version; the old 37-record initial review numbering must not be mixed in.

The assisted review shows that the NQ EM drop cannot be directly interpreted as an across-the-board semantic quality regression; the new approach has local repair signals pending confirmation, along with harmful changes that do not answer the question. The default strategy is still not promoted; first confirm the key cases, unify the answer-completeness criterion and the reference doubts, and then decide on the next version.

## Human confirmation of key cases (2026-10-14)

The user agreed overall per the [confirmation checklist](../../ANSWER_REPAIR_CONFIRMATION_20261014.md) (original wording "agree"), covering 28 key cases and their duplicate records; the remaining 97 unique samples are still assisted review conclusions pending batch confirmation. The confirmation method was confirmation of a chat-assisted summary, not an independent blind review; a separate independent blind second reading without seeing Codex's conclusions agreed with the assisted review on the 9 repairs and 3 harmful repairs. Record files: `data/annotations/human/answer_repair_v6r1_confirmations_20261014.json` (user confirmations and changed rulings), `data/annotations/human/answer_repair_v6r1_pairs_20261014.jsonl` (per-record pair_review, bound to the result SHA-256, output SHA-256 `2f0dc68b…`).

| Batch | Confirmed | Pending confirmation | Semantic repair | Semantic harmful repair |
|---|---:|---:|---:|---:|
| FEVER reviewed 14 | 3 | 11 | 3 | 0 |
| FEVER dev50 | 4 | 46 | 1 (duplicate of reviewed) | 2 |
| NQ reviewed 14 | 12 | 2 | 6 | 0 |
| NQ dev50 | 11 | 39 | 3 (all duplicates of reviewed) | 1 |
| Deduplicated | 26 | 97 | **9** | **3** |

2 deferred (#113 Little League pitching distance, #121 Soviet Union start/end years) are both reference mismatches and are not counted. Group C changed rulings: #31, #32 initially judged correct (not counted as repairs); #54 support recorded as insufficient; #109, #125 support recorded as yes. The 7 answer-completeness criteria adopted in the confirmation are in the confirmation checklist, and subsequent v7 paired reviews follow the same criteria.

Meaning of this set of human confirmations: v6r1's FEVER NLI has 3 real repairs on target error samples, along with 2 harmful repairs on dev50 caused by entity errors; NQ verbatim extraction has 6 real repairs that frozen EM did not recognize and 1 real harmful repair that EM did not recognize. v7's two changes target exactly the causes of these 3 harmful repairs; their effect is still subject to held-out evaluation.
