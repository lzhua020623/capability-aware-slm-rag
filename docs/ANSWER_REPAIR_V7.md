# E v7: separating short answers from citations, FEVER subject-entity constraint

> **Run (2026-10-14, Kaggle T4 x2)**: results in [ANSWER_REPAIR_RESULTS_20261014_V7.md](ANSWER_REPAIR_RESULTS_20261014_V7.md). The FEVER development batches match the offline replay, held-out 1 repair / 4 harmful repairs; NQ frozen EM is net negative or flat across batches, and significantly negative on held-out. Neither policy is promoted; this document is the pre-run protocol and is not revised after the results. Batch 4 `heldout3216` has also been run once: utilisation stratum EM 0→10, correct stratum 63→54, see the same results report.

Designed and frozen on 2026-10-14. v7 is the next version after the v6r1 assisted review, and the scope of changes is only two points: NQ splits the "answer" and the "supporting citation" into two fields; FEVER's NLI decision only lets windows that mention the claim's subject entity vote. Model, quantization, greedy, seed, NLI model version and the 0.90 threshold all carry over from v6, with no new model calls. This document is the pre-GPU-run protocol; results are written in a separate report, and this document is not revised.

## Why these two points

The 128-item assisted review of v6r1 (assistant + independent blind rating, the two reads agreeing) shows:

- The 15 frozen-EM regressions of NQ `extractive_reanswer` were all "correct answer but expanded into a full sentence", and another 6 semantic repairs were not recognized by EM; there was only 1 true semantic harmful repair (statute). The problem lies in the output form, not in the selection ability. dev50 had 24/50 invalid outputs, of which 17 were "non-verbatim" rejections caused by whitespace/quote formatting differences, and 5 were unescaped quotes inside JSON.
- The 2 harmful repairs of FEVER `fever_nli` (Men in Black II, Henry Cavill) were both triggered by high-scoring contradiction windows citing the wrong entity; among the 14 post-review items, 4 conflicts were also caused by windows from same-name/unrelated pages (Eva Green vs The Big Green, Alan Shepard vs Weinel/Kinnan, Babur vs Bidar Bakht).

## Pre-fixed policies

### `short_answer_quote` (NQ only)

Same Qwen2.5-7B-Instruct / NF4 / FP16 / greedy / seed 42, at most one 384-token generation, no JSON stopper. The model outputs three lines:

```
RANK: <evidence block number>
QUOTE: <a contiguous source span within that block, ≤60 words>
ANSWER: <the shortest answering span within QUOTE, ≤20 words>
```

Or the single line `INSUFFICIENT`. Program rules:

1. QUOTE must be localizable as a character slice of that block; ANSWER must be localizable as a character slice within QUOTE. Matching ignores whitespace and quote style (`` '' and curly quotes are normalized to "), keeps all words and punctuation, and slices must fall on word/punctuation boundaries. `Can't` can match the corpus's `Ca n't`, `York` cannot match `Yorkshire`, and negations cannot be deleted nor spans concatenated.
2. Final answer = the source slice of ANSWER, citation = the source slice of QUOTE; no model-authored text is returned.
3. When ANSWER and the initial answer are equal under frozen EM normalization the status is `kept`, otherwise `repaired`.
4. Lexical anchor: at least one third of the question's content words (≥3 letters, stopwords removed) must appear in the whole cited passage, otherwise the status is `unanchored_candidate` and the initial answer is kept. This is a weak rule that only blocks completely off-topic passages; whether it is useful must be shown by the with/without-anchor replay comparison on held-out and cannot be assumed.
5. Any failure of format, numbering, localization or length falls back to the initial answer and records `invalid_output`.

### `fever_nli_anchored` (FEVER only)

Same `microsoft/deberta-large-mnli` (revision `7296194b…`), same windowing, FP32, threshold 0.90, all window probabilities saved; classifier cost is exactly the same as v6. The only change is the decision rule:

- Take the claim's subject-entity block = the first run of capitalized words after removing a sentence-initial article/preposition (numbers do not count, a possessive s does not break it). For example "Henry Cavill starred in two Superman films." → `henry cavill`; "The 1974 crime film The Sugarland Express was…" → `sugarland express`; "In 2003, Eva Green debuted in film." → `eva green`.
- Only windows in which every word of the subject block appears in the window text (singular/plural allowed) are eligible to vote; scores of the other windows are recorded but ignored.
- The voting rule is the same as v6: among eligible windows, any entailment ≥ 0.90 with no contradiction ≥ 0.90 → SUPPORTS; the reverse → REFUTES; both or neither → keep the initial answer.
- Excluded high-scoring windows, each window's anchor coverage, and missing subject-block words are all written to `anchor_filter`, reproducible by replay.

Attribute words in the predicate are not required to appear (such as Australian in "Brubaker is a solely Australian film"), because refuting evidence would naturally give another value. When subject words appear in the evidence as aliases/pronouns they are conservatively excluded; this is intentional: exclusion can only keep the answer unchanged, never make it wrong.

### Offline replay (design-stage evidence, not held-out)

Only the decision rule is swapped, on the window probabilities already saved by v6r1 (`scripts/replay_fever_nli_anchored.py`, output in `artifacts/answer_repair/v7_design/`):

| Batch | n | Initially correct | v6 final | v7 final | v6 repaired/harmful | v7 repaired/harmful |
|---|---:|---:|---:|---:|---:|---:|
| FEVER 14 post-review items | 14 | 4 | 7 | 10 | 3/0 | 6/0 |
| FEVER original dev50 | 50 | 45 | 44 | 46 | 1/2 | 1/0 |

The 3 new repairs are the three conflicts of Eva Green, Alan Shepard and Babur being resolved; the 2 eliminated harmful repairs are Men in Black II and Henry Cavill. This rule was designed while looking at these samples, so the numbers above only show that the rule works as designed and cannot be taken as evidence of effect; the effect must be judged on the held-out below. The new NQ policy requires real generation and has no offline replay.

## Fixed data and run order

The experiment bundle is generated by `scripts/prepare_reviewed_answer_repair.py --experiment v7`, with manifest `configs/splits/answer_repair_reviewed_v7.json`.

1. `reviewed`: the 28 post-review training development samples from v5/v6 (FEVER 14, NQ 14), compared pairwise with v6r1.
2. `dev50`: v1's fixed dev50 × 2, to observe harmful repairs of correct answers, compared pairwise with v6r1.
3. `heldout`: all shared training samples E has never seen (Diagnosis v1/v2 training intersection − dev50 − the 37-item review bundle). FEVER 267, NQ 146. Sample selection does not read labels, correctness or reference answers. **Run only once, report honestly regardless of the outcome, and do not tune any threshold or prompt on it.**

The offline label composition of batch 3 (described only here; the script does not read it): FEVER 232 correct + 35 insufficiency, no utilisation labels; NQ 126 correct, 13 insufficiency, 7 accessibility, also no utilisation labels, but 75 of them are wrong under frozen EM. So this batch mainly answers "will E break samples that were misrouted (already correct)" and "will answers that EM judges wrong but are semantically correct be broken", and cannot answer the repair rate. NQ samples that truly carry utilisation labels and that E has not touched are in the 3216 extension pool of the Diagnosis v2 training set (1587 items, including 120 utilisation).
4. `heldout3216` (NQ only, separate experiment bundle): 300 items stratified by offline labels from the 1587 above: all 120 utilisation, all 35 accessibility, 100 correct, 45 insufficiency (seed 42 hash ordering, `configs/splits/answer_repair_nq_heldout_3216_v1.json`). Gold short answers are taken from the `gold_answer` column of the local workbook `NQ_C1_3216_合并全量.xlsx` provided by the Diagnosis owner (SHA-256 `a82961a6…`, not committed), exported as `data/annotations/gold/nq_heldout_3216_v1_gold.jsonl` (committed); the workbook's question and c1_prediction agree 300/300 with the upstream committed export, and `em_reference` agrees 300/300 with the recomputed EM. Frozen EM: 63 initially correct / 237 wrong. Experiment bundle `artifacts/answer_repair/v7_heldout3216/answer_repair_v7_heldout3216_gpu.tar.gz` (SHA-256 `6f27757935128f3e04e2881e1fb832885b10d45ef55316fa425b889950651bde`), notebook `notebooks/answer_repair_colab_v7_heldout3216.ipynb` / `answer_repair_kaggle_v7_heldout3216.ipynb`. **Likewise run only once**; results are reported separately for the four strata; this is a label-stratified held-out that does not represent the deployment distribution, and the repair rate is interpreted only within the utilisation stratum.

Costs are recorded separately per the v6 protocol: Qwen calls/generation seconds/input tokens count only NQ; NLI forward batches/classification seconds/input tokens count only FEVER.

## Reporting measures

- Frozen FEVER accuracy and NQ EM/F1 remain the main tables as before, reported per batch, without merging overlapping IDs.
- Paired semantic review follows the four items of v6r1 (initial/final correctness, initial/final support) and the unified completeness measure in `docs/ANSWER_REPAIR_CONFIRMATION_20261014.md`; assistant suggestions are not human ground truth.
- Must be reported together: the counts of `unanchored_candidate`, `invalid_output` and `nli_uncertain` and the proportion of originally wrong items among them; the number of high-scoring windows excluded by anchors; the replay comparison with/without the NQ anchor rule.
- If any batch shows a net negative gain or a held-out harmful-repair rate higher than dev50, write it into the results as is; the rules must not be changed afterwards and the same batch rerun.

## Code

| File | Contents |
|---|---|
| `src/compensation/anchors.py` | Subject-entity block, content-word anchors, coverage; no model |
| `src/compensation/nli_repair.py` | `repair_from_nli_trace_anchored`; v6 function kept as is for historical replay |
| `src/compensation/short_answer_repair.py` | Three-line format, whitespace/quote-normalized localization, short-answer validation |
| `scripts/replay_fever_nli_anchored.py` | Offline replay of the v7 FEVER rule using v6r1 traces |
| `scripts/run_answer_repair.py` | Registers the two policies; `fever_nli_anchored` resumption is validated by v7 rules |
| `tests/test_answer_repair_v7.py` | Anchors, decisions, localization, parsing, resumption, adapter boundaries |

```sh
python3.11 -m unittest discover -s tests
python3.11 scripts/prepare_reviewed_answer_repair.py --experiment v7 --manifest configs/splits/answer_repair_reviewed_v7.json --out-dir artifacts/answer_repair/v7_reviewed --bundle-name answer_repair_v7_reviewed_gpu
python3.11 scripts/build_reviewed_answer_repair_notebook.py --experiment v7 --bundle artifacts/answer_repair/v7_reviewed/answer_repair_v7_reviewed_gpu.tar.gz --out notebooks/answer_repair_colab_v7_reviewed.ipynb
python3.11 scripts/build_kaggle_answer_repair_notebook.py --notebook notebooks/answer_repair_colab_v7_reviewed.ipynb --bundle artifacts/answer_repair/v7_reviewed/answer_repair_v7_reviewed_gpu.tar.gz --out notebooks/answer_repair_kaggle_v7_reviewed.ipynb
# Batch 4 (requires the local workbook)
.venv/bin/python scripts/build_nq_heldout_from_diagnosis_inputs.py --workbook "data/annotations/NQ_C1_3216_合并全量.xlsx"
python3.11 scripts/prepare_nq_heldout3216_bundle.py --c1 artifacts/answer_repair/v7_reviewed/inputs_extra/nq_heldout3216_c1.jsonl
python3.11 scripts/build_kaggle_answer_repair_notebook.py --notebook notebooks/answer_repair_colab_v7_heldout3216.ipynb --bundle artifacts/answer_repair/v7_heldout3216/answer_repair_v7_heldout3216_gpu.tar.gz --out notebooks/answer_repair_kaggle_v7_heldout3216.ipynb
```
