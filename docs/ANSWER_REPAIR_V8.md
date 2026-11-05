# E v8: NQ original-answer protection and candidate verification

Development version of 2026-10-19, not yet promoted to default. The original 75-sample four-group experiment failed on the original-check format, and after r2 fixed the entry point the candidate format still did not pass. Now the [v8r3 real results](ANSWER_REPAIR_RESULTS_20261019_V8_R3.md) have completed 32 outputs and passed the frozen replay and engineering gates; the full pipeline changed only N96, EM 1→1; semantic wrong acceptances and wrong rejections still need optimization, and human review is not complete. Historical bundles and results are all kept.

## What this round tests

v7 can find real citations, but a real citation may belong to another work, another event or another relation, or may not answer the question. Direct replacement also harms aliases or short answers that were originally correct. This round tests: under the same model, same question and same C1 Top-5, whether first judging whether the original answer is really wrong, and then requiring the repair candidate to pass semantic verification, can reduce harmful repairs while keeping valid repairs.

At runtime only the question, initial answer, retrieved evidence and sample identifier of `DiagnosisInput` are read. The evidence is the Top-5 actually retrieved by C1, not oracle gold evidence. Reference answers, human labels and EM/F1 are used only for selecting the development queue and for offline reporting, and do not enter prompts or runtime branches.

## Fixed inputs and controls

Uses the [75-item finalized development list](../configs/splits/answer_repair_nq_train_confirmed_dev_v6.json): 11 internal E utilisation targets and 64 correct controls. The two partial answers N33/N62 are also targets. The 25 unresolved items remain excluded; the main source, C1 answers, gold and frozen scoring are not changed. Human confirmation used the assisted review in this conversation, not blind review or independent annotation.

This is a deliberately filtered NQ Train development queue and cannot estimate the overall error rate, system routing gain or unseen-set net improvement. These 75 items cannot be used for tuning and then called held-out. FEVER keeps its existing implementation in this round and passes regression tests; this round claims no new gain on FEVER.

| Control group | Treatment | Generation limit per item |
|---|---|---:|
| C1 | Reuse the original answer, no regeneration | 0 |
| `short_answer_quote` | v7 extracts the short answer and citation in one pass | 1 |
| `conservative_repair` | Original-answer verification → minimal repair → candidate verification | 3 |
| `conservative_no_original_gate` | Minimal repair → candidate verification | 2 |
| `conservative_no_candidate_gate` | Original-answer verification → minimal repair | 2 |

The three v8 groups use exactly the same repair prompt; the ablations only remove the corresponding verification. Each of the four groups has 75 items, 300 outputs in total, at most 600 extra generations. All use the frozen Qwen2.5-7B-Instruct revision `a09a35458c702b33eeacc393d103063234e8bc28`, NF4, FP16, greedy, seed 42; each call has at most 384 output tokens. v8's extra compute is higher than v7's; calls, input/output tokens and latency must be reported side by side, and an equal-budget comparison cannot be claimed.

The experiment uses `--all`, deliberately covering both error targets and correct controls to test repairs/harmful repairs. It bypasses Diagnosis/Controller; production integration still calls E only when the upstream final action is `generation_repair`, and other actions make zero calls and keep the original answer. No new routing threshold is set in E.

## Actual decision rules

1. **Original-answer verification**: six items check, respectively, entity/work/event, relation with its direction and conditions, answer type, necessary completeness, evidence support, and conflict. The model may only output yes/no/uncertain, plus one localizable citation and a brief reason. All six yes keeps the original answer; repair is entered only if there is a no and no uncertain; any uncertain or format/citation error keeps the original answer.
2. **Minimal repair**: combining the question, the full Top-5 and the original answer, extract the shortest contiguous evidence fragment that is still sufficient to answer the question. why/how/definition answers may be a clause or a sentence; avoid truncating a reason into an entity because of the 20-word limit. Candidates are at most 64 words, citations at most 100 words. Return `INSUFFICIENT` when one complete fragment cannot answer. No extra examples, sample-ID special cases or human answer injection.
3. **Candidate verification**: recheck whether the candidate itself answers the question, and recheck the full Top-5. Replacement happens only if all six are yes; any no/uncertain or format/citation error falls back to C1. The group without this step is only a development ablation.

Verification is performed by separate calls to the same Qwen, **neither an independent NLI nor an independent judge**. Errors may be correlated, and passing all six items cannot be taken as true correctness. Citation and answer localization are traceable checks; semantic judgements still have errors.

All stages use strict labelled line-by-line output, with the JSON stopper disabled and raw generations kept. Citations allow display differences in whitespace and PTB quotes, but must fall within a contiguous, word-bounded window of the original passage; entities are not fuzzily replaced, negations are not deleted and sentence fragments are not concatenated. The final candidate uses the source slice. If identical after normalization, the original C1 expression is kept; valid aliases, dates and nicknames are not forcibly rewritten, and full names or extra administrative levels are not uniformly required.

This version is still limited by contiguous extraction: cross-passage synthesis, multi-hop reasoning or the absence of an explicitly extractable statement may prevent repair. Original-answer verification that is too conservative may also miss real errors; these must be reported via per-stage failure counts, unrepaired targets and human review records.

No evidence returns zero calls. Format/citation/uncertainty are recordable fallbacks; backend exceptions such as CUDA, network and model loading still stop the run and keep completed records, to avoid counting infrastructure failures as the model "staying correct". Resumption can be validated within the same environment; changing model/code/environment requires separate outputs. Each v8 record saves stages, prompt hashes, raw outputs, parses, citation offsets and per-call costs, so decisions can be re-executed on CPU and the final results and costs verified.

## Code and running

- Core implementation: [conservative_repair.py](../src/compensation/conservative_repair.py).
- Policy dispatch: [answer_repair.py](../src/compensation/answer_repair.py); Controller adapter: [controller_adapter.py](../src/compensation/controller_adapter.py).
- Protocol: [answer_repair_v8_nq75_v1.json](../configs/answer_repair_v8_nq75_v1.json).
- Frozen packaging: `python scripts/prepare_answer_repair_v8.py`.
- Kaggle/Colab: [answer_repair_v8_nq75_transport_r1.ipynb](../notebooks/answer_repair_v8_nq75_transport_r1.ipynb), paired with `artifacts/answer_repair/v8_nq75/answer_repair_v8_nq75_gpu.tar.gz`.
- Single-group pre-check: `python scripts/run_answer_repair.py --dataset nq --c1 artifacts/answer_repair/nq_train_dev_v1/confirmed/through_batch06_c1.jsonl --all --strategy conservative_repair --check-only`.
- CPU replay entry point: `repair_from_v8_trace`; checkpoint resumption checks records automatically.

The Notebook validates the frozen code via the tar.gz and per-file SHA-256; it installs the same four direct dependency versions as recorded in the v7 cloud run, keeps the host CUDA PyTorch and exports the full environment. The four groups run in order, and samples cannot be selected or the policy changed based on intermediate answers; comparisons of randomness and latency across different cloud environments still need cautious interpretation.

## How to accept after output

`scripts/summarize_answer_repair_v8.py` generates `comparison.json` and `output_review_queue.jsonl`: raw EM/F1, per-group repairs/harmful repairs, per-stage pass/reject/invalid counts, calls and latency; additionally grouped by the 11 targets and 64 controls. Any new repaired answer keeps the pending human fields; first review the changes, then look at unrepaired targets, checking the question, initial answer, candidate and Top-5.

The old C1 human correctness does not automatically become the label of new answers. The human-reviewed net gain of new outputs is null before review. If only EM improves while semantic quality declines, report the decline; if EM does not match but human review approves, keep the frozen metric and report the semantic result separately.

This round only decides whether it is worth continuing: repair gains must exceed the harm to correct answers, and the extra compute must be acceptable; the default must not be replaced because of a positive gain on a small number of development samples. After that, fix the rules, confirm with the team an NQ/FEVER evaluation split not used for development, report full results and costs, and then decide whether to expand natural errors, adjust the verification method or adopt an independent verifier.

## Cloud execution fix (algorithm unchanged)

Version 1 actually obtained a T4 and failed after 30.5 seconds at the stage of reading the experiment bundle; model inference had not started and predictions were 0. Kaggle automatically extracted the uploaded archive, while the original Notebook only searched for the tar.gz filename. The transport r1 Notebook is now used, keeping the original bundle SHA-256 `906f096f78e36a4f272cdd863ce2e7383d62acb638c1e480a06c5d3f75daa7a3`.

For the auto-extracted directory, first compare the hash of the already fixed checksums document `5499fcca237c5d8f50187c27bf24f5f6189176dab8d86184b507116420f3997e`, then check all 85 files, and copy only the checked content into the run directory. 7 loading and replay tests and the extraction-layout verification of the real frozen bundle passed; the whole repository now has 136 tests passing. New and old Notebooks, failure records and loading verification are all kept; the model, the 75 samples, prompts, scoring and the four-group ablations were not adjusted.
