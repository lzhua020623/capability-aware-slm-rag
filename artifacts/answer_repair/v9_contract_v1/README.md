# v9 independent constrained output interface experiment

This is an engineering experiment on 12 brand-new fictional examples; it is not a formal NQ/FEVER evaluation, not model training, and does not reuse the retired smoke10/8. The original three v9 snapshots and the v7/r3/r4 packages all remain frozen; the new entry point will not promote the default strategy.

Reuses the v9 full Top-5, decomposition, isolated verification and strict post-check. The two strategies use prompt-only and XGrammar 0.2.8 decoding respectively, **both groups' prompts contain the same JSON Schema text**, to avoid treating prompt differences as decoding gains. Model/revision/NF4/seed/384 token budget unchanged, at most 144 calls and 48 final predictions. There is no direct re-answer group here, and it is not stitched together with old smoke numbers.

Generation constraints include the decision enum, legal evidence IDs, array length, FEVER repair labels and the original-answer constant when keep; duplicate IDs/questions, empty rationale, unchanged repair and NQ source-text location are still rejected by the old post-check. Valid JSON and a legitimate source cannot prove semantic support.

**Pre-GPU threshold**: all 48 present, frozen replay passes, budget not exceeded, at least 22 of the 24 constrained results pass the post-check, and Unknown ID and keep rewrites are 0. On failure, archive and do not tune prompts around this batch again; a pass only allows separately freezing a real semantic development run, and cannot be used to claim improved accuracy.

Locally, the fixed Qwen tokenizer (152064 vocabulary) was used to complete 36 native grammar compilations / positive-example acceptances, and 24 negative examples of wrong keep / illegal IDs were rejected; see [CPU verification](cpu_grammar_check.json). Qwen model weights were not loaded, **this is not a CUDA/NF4 compatibility or real generation pass**. 26 Python checks and frontend validation passed; the engineering examples will not be mixed into the 200-item real annotation quality review.

The GPU package and original notebook remain frozen. Kaggle [private task](https://www.kaggle.com/code/yearo111/e-contract-v1-synthetic-20261022) Version 1 (361060970) showed failure after running 573.2 seconds; all 71 raw outputs and the full log have been downloaded. **The 48 predictions and cloud replay are complete; the failure occurred only in the final ZIP export.** The original tar fixed source mtime to 0 (1970), and ZipFile in the notebook by default rejects files earlier than 1980, so the internal export left only a 22-byte empty ZIP. The full working directory was collected using Kaggle's Download output, and no results were lost.

Local Python3.13 frozen replay passed 48/48, the 40 source files and the checksum list are byte-identical to the input package, and the recomputed summary matches. 96 actual calls; the recorded output_tokens_estimate maximum was 109, and the maximum for constrained calls including the stop token was 110, all within the 144 calls / 384 token budget. The original ZIP SHA-256 is `864f9f3ca01b305be3434accb06c4b01ee55bdad320f5b88b88fa6aec687b1bd`.

| Strategy / decoding | Strict post-check compliant | Calls | Cumulative generation seconds |
|---|---:|---:|---:|
| evidence_id / prompt_only | 10/12 | 12 | 62.50 |
| evidence_id / XGrammar | 11/12 | 12 | 66.19 |
| factored_evidence_id / prompt_only | 10/12 | 36 | 126.93 |
| factored_evidence_id / XGrammar | 11/12 | 36 | 130.58 |

**The constrained groups were 22/24 compliant, with Unknown ID and keep rewrites both 0, meeting this batch's pre-declared engineering threshold.** prompt_only was 20/24; the newly compliant items are the same S02 no longer rewriting the keep answer under the two strategies, which cannot be seen as gains on two independent questions. The remaining two constrained rejections both belong to S03: the model gave the digit 9 while the selected source text says nine, which is still rejected by the existing verbatim snippet rule. Compliant results include keep, missing evidence and conflict; the compliance rate is not answer accuracy. The decomposition groups used 36 calls this time, and cannot be called compute-matched with the 12-call single-round groups.

Local replay is not new generation; the 12 fictional questions of this batch have no real benchmark gain or human review conclusion. Only a separately frozen real semantic development protocol is allowed next; the strategy is not promoted by default, this batch is not reused to tune prompts / relax the source-text rule, and the retirement requirement for the old smoke10 is not lifted. Next, label quality review continues, and then wrong answers wrongly accepted, correct answers wrongly rejected, fixes / harmful repairs and cost are checked on trustworthy development labels.

Raw files are in [kaggle_20261022](kaggle_20261022/), [full collection and replay receipt](result_validation.json), [recomputed summary](release_audit/recomputed_summary.json), [local verification code](release_audit/verify_export.py). The full output ZIP is `kaggle_outputs_v1_361060970.zip`; the original empty inner ZIP is still contained in it. The collection process did not rewrite the original outputs, gold, human labels or model.

A separate `notebooks/answer_repair_contract_v1_exportfix.ipynb` has been added, which only sets ZipFile to `strict_timestamps=False`; the original notebook, GPU tar and inference code are unchanged. The regression test uses the actual notebook's finally code to reproduce the 1970 timestamp error, then verifies that the fixed ZIP keeps the same paths and contents and skips pycache. See [export fix receipt](bundle_manifest_exportfix.json). This variant was not submitted for a GPU run: the existing 48 items are already complete and do not need a rerun.

Files: `synthetic_cases.jsonl`, `preflight.json`, `cpu_grammar_check.json`, `bundle_manifest.json`, `answer_repair_contract_v1_gpu.tar.gz`. The notebook is at `notebooks/answer_repair_contract_v1.ipynb`. The package records hashes of 40 source files; the raw model generations, environment and export results are archived separately after being fully collected.

```sh
python scripts/run_answer_repair_contract.py --check-only --out artifacts/answer_repair/v9_contract_v1/preflight.json
python scripts/run_answer_repair_contract.py --compile-only --out /tmp/contract_cpu.json
# The following require CUDA, the project's pinned dependencies and xgrammar==0.2.8
python scripts/run_answer_repair_contract.py --out results/answer_repair/contract_v1/predictions.jsonl
python scripts/run_answer_repair_contract.py --replay --out results/answer_repair/contract_v1/predictions.jsonl
# Collect and replay offline with the same Python major/minor as the cloud: 3.13 here, no GPU needed.
python3.13 artifacts/answer_repair/v9_contract_v1/release_audit/verify_export.py
# Only generate the export-fix variant, keeping the original notebook; this batch does not need a model rerun.
python scripts/build_answer_repair_contract_notebook.py --export-fix
```

Uses the [official XGrammar HF integration](https://xgrammar.mlc.ai/docs/latest/start/quick_start.html). New code and the frozen package are in `src/compensation/constrained_evidence_id.py`, `qwen_constrained.py` and `scripts/run_answer_repair_contract.py`. The insufficiency of the original interface semantics still needs verification on real data, and the local grammar pass does not lift the old v9 stopping rule.
