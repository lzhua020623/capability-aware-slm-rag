# E v8r2: fix the output contract first, then run a real-model smoke

This is an engineering revision of the original v8. The original 906f096f input bundle, the 0bf8aa78 result ZIP and the strict replay remain unchanged; results from re-parsing old outputs with relaxed parsing are not passed off as a new model experiment.

## Changes and boundaries

- **Stop repeated output during generation**: only v8 requests labelled generation. Stopping is triggered after the REASON line of the nine-line check and the ANSWER line of the three-line candidate are ended by a newline; generation is not truncated early while the last field is not yet finished. EOS can normally end a compliant output without a trailing newline. The program saves the full actual generation and records whether the stopper was triggered; it does not pick one post hoc from repeated, contradictory or multiple sets of checks.
- **Explicit single check**: the prompt requires one overall judgement based on the full Top-5, with no second per-passage set of checks appended after stopping. The three-stage semantic criteria, branches and call budget are unchanged.
- **Number representation**: v8 accepts one integer or one level of square brackets, such as `3`, `[3]`; it does not accept multiple numbers, explanatory text, negative numbers or nested brackets. rank=0 still only allows uncertainty with QUOTE=NONE.
- **Citation enclosing marks**: the original string is matched first; if it does not match, only removing one pair of matching outer quotes is tried, followed by the existing contiguous-fragment and word-boundary localization. The final citation/answer still returns the source slice of the evidence. Negations cannot be deleted, entities replaced, text rewritten, concatenated or approximately matched. Quotes already present in the evidence are preferentially kept.
- v7's three-line parsing, citation constraints and generation method keep their original behaviour. The labelled stopping newly added to the shared backend is only called explicitly by v8. Model revision, NF4/FP16, greedy, seed and the 384-token limit per call are kept.

Code and prompts switch to the new version `v8r2_single_response_grounded_display_v1`; the fingerprint includes the new stopping module, and resuming old outputs with the new code is refused. Stage traces add the output contract and stop observation, and still support CPU replay of the saved complete three stages. A stop observation of false only means the output was not ended by this stopper; it may be EOS or the token limit, and cannot automatically be treated as a format error.

## Real-failure regressions and offline diagnosis

`tests/fixtures/answer_repair_v8r1_contract_cases.json` fixes three classes of real failures from the original result ZIP, containing only runtime evidence, original generations and source hashes. Tests cover stopping during generation, rejection of full historical repeated responses, enclosing quotes and number compatibility, as well as entity, negation, word-boundary and source-slice constraints. Three-stage backend calls, trace tamper-proofing and CPU replay are also verified.

The offline parsing check only measures whether existing text conforms to the new parsing contract. The subsequent model outputs needed after branches change do not yet exist, so new final answers or semantic improvements cannot be computed from it. The real-model smoke is stored separately and not confused with offline tests.

## Fixed 8-item smoke

From the 75 already developed on, IDs N1/N2/N8/N12/N15/N31/N48/N96: 4 confirmed correct controls and 4 E targets. Chosen to cover repetition, quotes, numbering, work identity, comparison relations and reason explanations; it is not an unseen evaluation and is not used to prove gains.

Protocol in `configs/answer_repair_v8r2_smoke8_v1.json`. Four groups of 8 each, 32 predictions in total, at most 64 generations. At runtime only the C1 question, original answer and actual retrieved Top-5 are still used; gold and the initial review are reported only offline.

Pre-fixed engineering pass conditions: 32 records complete and in correct order; all stages actually entered by v8 are valid; at least one valid call for each of the three stage types; at least one complete three-stage path; backend stop observation present; generated text not truncated post hoc; no more than 64 generations. v7's existing strict parsing errors are reported separately, and the control group is not modified because of them. An engineering pass does not require an EM improvement and does not grant semantic correctness labels.

Preparation entry point: `python scripts/prepare_answer_repair_v8_r2.py`. The new bundle uses the validated loader that supports Kaggle auto-extraction. The Notebook only runs the 8-item smoke and exports the results and `smoke_acceptance.json`, and will not automatically launch the full experiment.

First check the real smoke's format, citations, stage coverage and cost. If it passes, rerun the same 75-item four-group development experiment with another frozen bundle; if not, save the failure and continue fixing. Humans may review the original 15 changed answers and the new outputs in parallel, without waiting for the 25 old unresolved items to be finalized. At no stage is this development queue treated as held-out, or Qwen self-checking treated as an independent judge.

## Real execution results

Kaggle version 1 ran successfully in 585.8 seconds, with 32 predictions and 41 generations, and the frozen CPU replay passed. The entry check recovered, but 6 candidate format errors and 3 source rejections made r2's original pre-set gate fail, and the full pipeline had no changed answers. See [results and semantic issues](ANSWER_REPAIR_RESULTS_20261019_V8_R2.md). The subsequent [v8r3](ANSWER_REPAIR_V8_R3.md) is a separate new revision and does not overwrite this version's frozen bundle or failure records.
