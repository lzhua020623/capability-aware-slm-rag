# FEVER500 7B C0 / C3 results

The frozen FEVER C0 and C3 runs are complete. Both conditions contain the same 500 unique sample IDs in manifest order, with 250 SUPPORTS and 250 REFUTES. No samples were resampled or excluded.

## Accuracy

| Condition | Correct / total | Accuracy | SUPPORTS | REFUTES | UNKNOWN |
|---|---:|---:|---:|---:|---:|
| C0: No Evidence | 360 / 500 | 72.0% | 121 / 250 (48.4%) | 239 / 250 (95.6%) | 0 |
| C3: Oracle Evidence | 476 / 500 | 95.2% | 230 / 250 (92.0%) | 246 / 250 (98.4%) | 0 |

Oracle evidence increases accuracy by 23.2 percentage points on this frozen subset. The improvement is concentrated in SUPPORTS samples; the C0 accuracy gap between labels is substantial.

## Paired outcomes

| C0 | C3 | Samples |
|---|---|---:|
| Correct | Correct | 357 |
| Wrong | Correct | 119 |
| Correct | Wrong | 3 |
| Wrong | Wrong | 21 |

These are C0-to-C3 comparisons, not the protocol's Recoverable Failure Rate. That metric requires C1 wrong / C3 correct on matching IDs. Likewise, the 24 C3 errors are not yet the 32B routing set: first intersect them with C1 errors. This branch does not contain the FEVER C1 results.

## Protocol and artifacts

- Model: `Qwen/Qwen2.5-7B-Instruct`, 4-bit NF4, greedy, `max_new_tokens=64`, seed 42.
- Config: [preliminary.yaml](../../configs/preliminary.yaml).
- Manifest: [preliminary_fever_500.json](../../configs/splits/preliminary_fever_500.json).
- Source data MD5: `4b1287199b1bbefa1529f59283913b87` (`data/processed/retrieval/fever_dev.jsonl`, not tracked).
- Inference/evaluation code and input fingerprint matches the local runner at commit `4edc319d5135fb9a74737d00f3e2bc5ecb364065`.
- Experiment fingerprint: `ec116ab4ea02921b704cc77c5b9ff8e782220cc351db21002133160b4f81ae95`.
- All 1000 result records report `cpu_offload=false`.
- Original result files: [C0](fever_c0_7b.jsonl), [C3](fever_c3_7b.jsonl).
- Original server log: [fever_run.log](../../fever_run.log).

Downloaded files were copied byte-for-byte. SHA-256 checksums:

```text
551731d6c3ba5444bdd5c736ec5c07427ac23fed87fc4fe1c76f91e4dc6230dd  results/preliminary/fever_c0_7b.jsonl
e8bdc18d2e806e3edcf4705b9c9ba8659f3cb433fa31dd57b6c22fb492280753  results/preliminary/fever_c3_7b.jsonl
2cab0881697e1c92c249bee9b48a7030c6960231840395f1e9ae3d0ebda9eb1f  fever_run.log
```

## Validation

The archived records were checked against the frozen manifest and local source data: unique IDs, labels, claim text, empty C0 evidence, exact C3 gold evidence, model, experiment fingerprint, raw-output parsing, correctness and finite nonnegative latency. Every one of the 1000 sequential log entries agrees with its result record, including latency rounded to two decimals.

`python scripts/run_fever_preliminary.py --check-only` passes with the source data installed. Running the default command reports that both conditions are already complete and reproduces the accuracy summaries without loading the model.

The archived files do not include a package-version snapshot or a pinned Hugging Face model revision, so they do not fully specify the server software environment.
