# Colab and Kaggle experiments for E

The current working basis is the completed v7; the next round, v8, is not yet implemented. The v7 reproduction entry points are kept below; the run history and old environment status of historical v1 to v6r1 are in the [historical execution records](history/answer_repair/GPU_RUNS_V1_V6R1.md). Existing results need no re-inference; offline replay is preferred.

## Completed v7 experiments

| Experiment | Notebook | Inputs and results |
|---|---|---|
| Main bundle: review set, dev50, pilot-pool held-out set, 541 in total | [Colab](../notebooks/answer_repair_colab_v7_reviewed.ipynb) / [Kaggle](../notebooks/answer_repair_kaggle_v7_reviewed.ipynb) | `artifacts/answer_repair/v7_reviewed/` |
| Batch 4: NQ3216 stratified held-out set, 300 | [Colab](../notebooks/answer_repair_colab_v7_heldout3216.ipynb) / [Kaggle](../notebooks/answer_repair_kaggle_v7_heldout3216.ipynb) | `artifacts/answer_repair/v7_heldout3216/` |

Both batches have been run once on Kaggle. The experimental design is in the [frozen protocol](ANSWER_REPAIR_V7.md), and the complete results, limitations and follow-up roadmap are in the [v7 report](ANSWER_REPAIR_RESULTS_20261014_V7.md). The held-out sets are not used as the tuning development set for the next round, and reusing a Notebook cannot be recorded as a new algorithm version.

Each version directory keeps the input `.tar.gz`, the raw output ZIP, `kaggle_20261014/` and `release_audit/`. Exact hashes are authoritative in the [archive manifest](../artifacts/answer_repair/git_archive_manifest.json); the extracted copy `gpu_bundle/` can be rebuilt when needed and is not committed.

## Offline replay without a GPU

Run from the repository root: extract the input bundle into a new temporary directory, then use the frozen code inside the bundle to check the saved results:

```sh
replay_dir="$(mktemp -d)"
tar -xzf artifacts/answer_repair/v7_reviewed/answer_repair_v7_reviewed_gpu.tar.gz -C "$replay_dir"
python3.11 scripts/audit_answer_repair_release.py \
  --bundle-dir "$replay_dir/answer_repair_v7_reviewed_gpu" \
  --results-dir artifacts/answer_repair/v7_reviewed/kaggle_20261014 \
  --out-dir "$replay_dir/audit"
```

Batch 4 uses the corresponding `v7_heldout3216` bundle and results directory, outputting to a different temporary directory. Replay checks the frozen source files, the results manifest, verdicts and scores; it does not load the model and does not automatically fill in semantic human labels.

Current human review uniformly uses the [v7 combined sheet](../artifacts/answer_repair/v7_combined_review/review_all.txt), with structured data in `review_all.jsonl` in the same directory; newly generated empty review queues do not overwrite existing human worksheets.

## Future new experiments

First complete new strategy development, fixed development samples, controls and the call budget, then generate a separate experiment bundle and Notebook. Use a new output directory and do not overwrite old bundles. The default output of the historical build scripts already points to the historical Notebook directory; new experiments must specify the version and output location explicitly. This round of repository tidying did not change the model, prompts, thresholds, strategy or samples.
