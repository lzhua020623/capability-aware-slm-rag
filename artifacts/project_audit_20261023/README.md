This directory stores the cross-branch and data audit of 2026-10-23. For conclusions and follow-up design see [Project audit and RSI plan](../../docs/PROJECT_AUDIT_AND_RSI_PLAN_20261023.md).

- `manifest.json`: source commits, check scope, environment, logs and artifact SHA-256.
- `branches.json`: commits, file lists and differences relative to main for 10 remote branches and 4 local branches.
- `preliminary_artifact_inventory.json`: Git blob consistency of cross-branch preliminary experiment files.
- `nq_delivery_audit.json`: structure, source, duplicate and split checks of the 74,097 NQ Train deliveries and the standardized initial labels.
- `cross_branch_split_audit.json`: cross-check of IDs and normalized questions against the published Diagnosis NQ splits.
- `split_variant_groups.json`: 17 groups of cross-split Unicode/punctuation variants, 35 IDs; contains no question text or answers; the original split was not modified.
- `*_tests*.log`: raw test logs, including the initial environment problems and the reruns in the correct environment. Finally 435 tests passed; this is not a GPU effectiveness experiment.
- `audit_data.py`: read-only data scanning script that depends only on the Python standard library; it requires the received local raw shards, label layers and partition files.

When rerunning the data checks, use a new output directory and do not overwrite this audit snapshot:

```sh
python artifacts/project_audit_20261023/audit_data.py /absolute/path/to/capability-aware-slm-rag /tmp/a-new-data-audit-output
```

This audit did not modify raw data, the knowledge base, indexes, labels, partitions or algorithms. The full NQ corpus is not at the configured local path, and the FEVER corpus is a cloud-drive placeholder; semantic correctness, full-store integrity, page/semantic near-duplicate isolation and pretraining contamination are outside the scope of the checks claimed as passed.
