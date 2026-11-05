# FEVER C1 900 Collection Complete · 2026-10-24

Kaggle Version #3 (361681141) completed successfully and has been downloaded and accepted. `prefill_chunk_size=256` is used uniformly; this directory does not splice in the 645 old outputs from Version #2.

| Purpose | Completed / Planned |
|---|---:|
| development | 200 / 200 |
| selection | 200 / 200 |
| final_candidate | 500 / 500 |

All 900 match the frozen questions, full evidence text, order, and raw generation records, with no missing or duplicate rows. The 77 files of the input archive, the 11 files of the output index, and the result ZIP hash printed in the cloud all checked out; the file bytes in the outer download archive and the inner result ZIP are identical. `completion_receipt.json` and `coverage_audit.json` store the acceptance figures; `archive_audit.json` stores the archive verification chain.

The three fictional engineering input checks passed; the long input was actually **13,139 tokens**, with a generation time of **51.50 seconds**. The real input `81950` (12,911 tokens) that previously hit OOM has completed; the measured input token counts of all 900 match the original frozen tokenizer length list. The actual environment is Qwen2.5-7B-Instruct at a fixed revision, 196 NF4 modules, FP16 compute, double quant; parameters are on `cuda:0`. The full environment and raw engineering trace are in `raw_outputs/environment.json` and `raw_outputs/engineering_smoke.json`.

The `generation_config` in the environment stores the model defaults; the actual calls are explicitly overridden by the frozen source to `do_sample=False`, 64 output tokens, seed 42, and 256-token prefill. The `do_sample=True` in the default config does not mean this round used random sampling; the effective call specification is recorded separately in the completion receipt.

The total successful run duration was **3,224.4 seconds (53 minutes 44 seconds)**. The 900 formal C1 generations took 3,000.29 seconds in total, averaging 3.334 seconds/question; input totaled 1,198,856 tokens, and output re-tokenization is estimated at 8,125 tokens (excluding EOS). This measure excludes CPU Top5 retrieval. The Kaggle runtime of the three attempts totaled 5,086.0 seconds, including two failures; the last successful run alone cannot represent the full trial-run cost.

All 900 can be parsed by the original FEVER parser: 795 are pure labels, 105 carry extra text, and no output contains two different labels at once. The extra text is saved as-is; the prompt is not modified based on it, and it is not directly counted as a QA error. Gold was not read, accuracy was not computed, and final candidate answers were not displayed or reviewed per question; what this round completed is the C1 baseline input, which does not constitute a gain conclusion for E or the Controller.

The Controller can use `raw_outputs/<role>/runtime.jsonl`; source and input fingerprints are pinned by the parent-level collection. The repository entry is `configs/controller/fever_c1_completed_v1.json`, which registers only the runtime and acceptance hashes; gold stays in the original offline directory.

`kaggle_version3_outputs.zip` stores the complete raw download, containing the cloud-exported `controller_fever_c1_v1_outputs.zip`; the unpacked code copy of the experiment package is not committed again. The raw run files are in `raw_outputs/`, and the raw cloud log is `kaggle_version3.log`.

Before the subsequent formal comparison, T3's per-group interfaces, budgets, evaluators, and execution seal still need to be completed, and the generation configuration unified. This directory does not rewrite the historical T2 manifest as fully ready.

Full CPU regression: **744 tests passed**, the original T2's **552 source hash checks passed**, and logs and credentials are in `validation/`. In addition, there are 7 completion-artifact regression tests, including one requiring acceptance to fail after a trace is deleted.
