# Frozen split slot

## NQ

Place the **original** frozen file here (do not regenerate IDs):

`diagnosis_nq_train_74097_split.json`

Expected: `splits.nq.test` length **11114**.

Until that file is present, the runner reports `FROZEN_SPLIT_MISSING` and will not run formal Test inference.

## FEVER

Frozen FEVER Test (do not reshuffle):

`fever_dev_13332_split.json`

Expected: `splits.fever.test` length **13332** (SUPPORTS=6666, REFUTES=6666), aligned with `baselines/data/fever_dev_13332.jsonl`.
