# E v5: reviewed development set and same-model semantic review

2026-10-14: the 37 chat-assisted review opinions have been confirmed, comprising 19 utilisation core candidates, 9 correct controls, and 9 deferred cases. The fixed development set includes only the first two groups, 28 records in total: FEVER 10+4, NQ 9+5. All come from the frozen C1 training intersection of Diagnosis v1/v2, excluding all validation/test IDs of both versions. The samples have been used for human-assisted review and strategy development and cannot serve as an unseen test set or represent the overall deployment distribution.

This round completed 112 real predictions on Colab T4 and passed CPU deterministic replay. The v5 candidates and same-model review did not increase the frozen correct counts on FEVER/NQ; see `ANSWER_REPAIR_RESULTS_20261014.md` for detailed results. v5 is not promoted to the default strategy, and the old experiment package is kept as is.

## Experimental questions

1. Can replacing the overemphasis on the shortest span with "the shortest but complete answer" reduce off-topic answers, missing entities, and misunderstood event relations?
2. After a candidate answer passes citation checking, can having the same Qwen 7B check the question, entity, relation, coverage, and evidence support reduce harmful repairs? What are its extra cost and the cost of rejecting correct repairs?

The second step is a second call to the same model, not an independent NLI verifier or human adjudication. The same model may repeat its mistakes. All outputs continue to be marked `independent_support_status=not_checked`; the model claiming a check passed does not count as evidence of correctness.

## Fixed comparison

The four conditions use exactly the same 28 queries, initial answers, and original-order Top-5, with Qwen2.5-7B-Instruct revision `a09a35458c702b33eeacc393d103063234e8bc28`, NF4, FP16, double quant, greedy, seed 42.

| strategy | Behaviour | Max calls per sample | Output token cap |
|---|---|---:|---:|
| `grounded_reanswer` | Keeps the v4 candidate prompt and parsing behaviour | 1 | 384 |
| `query_grounded` | New candidate prompt that fully answers the question | 1 | 384 |
| `guarded_reanswer` | Same candidate as the previous row, plus semantic review | 2 | 384 + 192 |
| `reanswer` | Re-answers directly with the original C1 instruction | 1 | 576 |

The full set has at most 140 calls and 112 final predictions. Review is not called when the candidate is invalid or abstains. Review also covers candidates identical to the initial answer. The four conditions are not four independent sample batches. The 576-token Reanswer only controls the total output budget cap; call counts, input lengths, and actual tokens/latency still differ, and cost must be reported truthfully. `query_grounded` and `guarded_reanswer` are used to isolate the effect of review; the notebook checks whether the raw candidate outputs of the two groups are identical, and if they differ, the final difference cannot be entirely attributed to review.

This changes the single-call experiment budget of v4 and is recorded as an explicit new version. The Controller default strategy and upstream Diagnosis/Probe thresholds are unchanged; in this round all samples execute E, which is an operation-level development ablation with samples selected by human category, not an evaluation of deployment routing gains.

## Implementation

- `src/compensation/answer_repair.py`: the new candidate prompt specifies answer type, full coverage, entity, time, and predicate relation. The initial answer, human labels, and reference answers do not enter the candidate prompt.
- `src/compensation/support_check.py`: the second call sees only the query, full evidence, candidate, and citations, and checks `query_match/entity_match/relation_match/coverage/evidence_support/conflict_free`. The candidate is accepted only if all six are `yes`; `no`, `uncertain`, and format errors all keep the initial answer. Failure does not mean the initial answer has been proven correct.
- v5 NQ answer span matching ignores spaces between words and punctuation, but all words and punctuation must match contiguously. For example, `1922-1991` can match `1922 - 1991`; negations must not be deleted, spans must not be spliced, and part of a word must not be treated as a whole word. The old v4 behaviour is kept.
- Outputs save the candidate, the raw review output and check items, two-stage tokens/latency, and total calls. Model exceptions continue to be raised, to avoid disguising them as keep. Interrupted unfinished samples are re-run; the cost of completed records excludes failed attempts not written to disk; run logs must be saved with the results.

These checks do not guarantee that the model has semantic judgement ability. CPU tests only verify the reject/accept logic, format contract, information isolation, and cost recording; model quality requires real outputs and human re-review.

## Samples and scoring

The fixed list is in `configs/splits/answer_repair_reviewed_dev_v1.json`, which keeps inclusion/deferral reasons, overlap with the old dev50, source commit/hash, and IDs where human conclusions disagree with frozen metrics. The preparer verifies the review package content hash, human record matching, complete confirmation status, training split, and consistency of C1 query/answer/evidence/reference answer.

The original source diagnosis, C1 predictions, and gold are not rewritten. Record 23 pharmacy school white coat, record 34 Columbia, and record 37 bias definition have been confirmed usable as correct controls, but frozen EM still judges them wrong. NQ has 5 human-confirmed correct controls, of which the old EM recognizes only 2; the 4 FEVER correct controls agree with the original label metric. These reference answer or wording differences must not be treated as semantic repair success or failure.

Results are reported in two layers:

1. Raw FEVER accuracy and NQ EM/F1, reported separately for all samples, the core group, and the correct-control group, with counts of invalid, abstain, review rejections, and calls/tokens/latency.
2. A separately generated before/after answer paired review table. Correctness and evidence support of the initial and final answers are checked by the same standard; all human result columns are initially empty. Until human re-review is complete, semantic repair counts and harmful repair rate show `null`, and the model review's `yes` is not automatically treated as human ground truth.

Review includes unchanged answers, keep, abstain, invalid, and review rejections; one must not check only results that appear to have been fixed. Human scores are bound to a specific result SHA-256 to prevent applying them to another output version. Templates and summaries do not silently overwrite existing files; summaries after re-review use a new `--out-prefix`.

The 9 deferred cases are kept in the manifest, awaiting a unified team definition; this round does not use these unresolved labels as scoring ground truth. The existing 19+9 records are chat-assisted confirmations, not independent blind reviews.

## Reproduction and execution

Local preparation and CPU checks:

```sh
python3.11 -m unittest discover -s tests -v
python3.11 scripts/prepare_reviewed_answer_repair.py
python3.11 scripts/build_reviewed_answer_repair_notebook.py
```

Experiment package: `artifacts/answer_repair/v5_reviewed/answer_repair_v5_reviewed_gpu.tar.gz`.
Notebook: `notebooks/history/answer_repair/answer_repair_colab_v5_reviewed.ipynb`.
On Colab, upload the notebook and experiment package, choose T4, and execute in order. Kaggle can import the notebook and add the package as a private input. No clone, retrieval rebuild, or raw data download is needed; the first model load still needs to download model weights.

The notebook verifies the archive and per-file hashes, runs environment checks and CPU tests, pre-checks the eight dataset/strategy commands, then executes the full development experiment and exports a ZIP. It does not automatically mount Drive, nor upload the repository or commit code. Completed results can be resumed in the same environment.

Single-group commands (in the extracted package root):

```sh
python scripts/run_answer_repair.py --dataset nq --c1 inputs/nq_reviewed_c1.jsonl --all --strategy guarded_reanswer --max-new-tokens 384 --out outputs/nq_guarded_reanswer.jsonl
python scripts/summarize_reviewed_answer_repair.py --manifest manifest.json --c1 inputs/nq_reviewed_c1.jsonl --results outputs/nq_guarded_reanswer.jsonl --out-prefix outputs/nq_guarded_reanswer_reviewed
```

Subsequently, real results decide whether to keep the new candidate prompt and review. First look at trustworthy repairs on the 19 error samples, harmful repairs on the 9 correct controls, and total cost, then return to the frozen Controller to check routing coverage; the small sample is not used to claim overall system effectiveness.
