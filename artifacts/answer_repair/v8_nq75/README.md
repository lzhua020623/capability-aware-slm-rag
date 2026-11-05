# v8: 75-item NQ Train development comparison experiment

This is the actually modified v8 development package: original-answer verification → minimal repair → candidate verification → keep the original answer when uncertain. Version 2 has completed 300 real outputs on T4, taking 74.7 minutes; in the full v8, all 75 original-answer verifications were invalid and fell back, none entered repair, and the strategy has not yet been promoted to default. 15 changed-answer records are pending human confirmation. See [results and failure analysis](../../../docs/ANSWER_REPAIR_RESULTS_20261019_V8.md) and `cloud_status.json`. The loading-failure record of version 1 is retained.

- Fixed 75 items per group: 11 internal E targets, 64 correct controls; the 25 pending items are excluded.
- Four groups: v7, full v8, without original-answer verification, without candidate verification. C1 reuses the original answers as a zero-call baseline.
- Expected 300 outputs, at most 600 Qwen generations. Call budgets differ, so the actual cost must be reported.
- `answer_repair_v8_nq75_gpu.tar.gz`: frozen code, inputs, protocol and offline initial review grouping.
- `manifest.json` / `preparation.json`: queue, strategy, hashes, expected outputs and call budget.
- `validation.json`: 96 portable tests run after extracting from the archive, and CPU prechecks of the four 75-item groups; does not include model inference results.
- `gpu_bundle/`: local extracted copy, not committed per the rules.

The run uses the transport r1 Notebook: `notebooks/answer_repair_v8_nq75_transport_r1.ipynb`. For the full rules see [v8 notes](../../../docs/ANSWER_REPAIR_V8.md). `kaggle_20261019/` saves the original 27 files; `answer_repair_v8_nq75_outputs_export.zip` keeps the original download. `release_audit/` saves the frozen decision replay and re-aggregation, and `result_validation.json` records hashes and the cross-environment F1 rounding tolerance. `review/changed_outputs_pending.jsonl` extracts the 15 changed-answer records; the original 300-item review queue remains pending.

Package SHA-256: `906f096f78e36a4f272cdd863ce2e7383d62acb638c1e480a06c5d3f75daa7a3`.

This is a development experiment; old C1 human labels cannot vouch for new answers, and the gain on 75 items cannot be taken as an unseen evaluation conclusion. The preparation and manifest in this directory are pre-run snapshots; the cloud execution status is saved separately, and they cannot be changed to "accepted".

## Kaggle loading fix r1

Version 1 obtained a T4 and ran for 30.5 seconds, then exited before model inference due to an archive lookup failure, 0 predictions. Kaggle had automatically extracted the tar.gz into a directory. `kaggle_v1_failure.json` keeps the failure record.

The current run entry is changed to `notebooks/answer_repair_v8_nq75_transport_r1.ipynb`, which can load the full archive or the auto-extracted directory; for the latter it uses the fixed checksums document hashes to verify all 85 valid payload files, and copies only these files to a writable directory. The original package, the 75 inputs, model, prompts and all experiment parameters are unchanged. `transport_r1.json` pins the hashes of the new Notebook and loader; `transport_validation.json` records the loading validation on the original package's real extracted layout.
