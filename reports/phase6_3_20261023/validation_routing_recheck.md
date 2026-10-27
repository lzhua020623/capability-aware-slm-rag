# Phase 5-7 validation routing recount, 2026-10-24

Recomputed independently using only these Git blobs at
`338ff34b039ed367220b0b916b48c4f7fc715e20`:

- `results/controller/controller_v1_validation.json`: final per-sample `action`.
- `results/diagnosis/diagnosis_v2_expanded_validation.json`: `target_label`, joined by `sample_id`.
- `results/probe/probe_v2_expanded_validation.json`: Prompt Development exclusion flag and target-label cross-check.

The script checks uniqueness, complete membership, dataset agreement, and
target-label agreement across all three validation sources before counting.
No test files are read. Counts come from the joined records; expected counts
serve only as a stop-on-mismatch gate before weight computation/export.

NQ validation: 406. Excluded Prompt Development sample:
`9148233787183514779`. Eligible: 405. Final action
`retrieval_enhancement`: 49.

| Offline target label | Recount | Weight |
|---|---:|---:|
| correct | 19 | 19/49 = 38.7755102041% |
| evidence_insufficiency | 23 | 23/49 = 46.9387755102% |
| evidence_accessibility | 0 | 0/49 = 0% |
| evidence_utilisation | 7 | 7/49 = 14.2857142857% |

Matches the prior Phase 5-7 result and the weights already used by the Phase 6-3
report, including all three source SHA256 hashes. Reweighting the saved label
metrics gives the same EM delta +3.8996339521 and F1 delta +4.9791792005
percentage points. The existing confidence intervals and conclusions are unchanged.

Saved together in the project's sibling outputs directory:

`../../outputs/phase6_3_validation_routing_20261024/`

- `recompute_phase5_7_routing.py`: standalone standard-library calculation script.
- `routing_weights.json`: counts, exact floating-point weights, excluded ID, source commit and hashes.
- `routed_samples.json`: all 49 joined sample IDs, final actions and offline labels.
- `checksums.json`: checksums of the script and both data files.

The script is also tracked at `scripts/recompute_phase5_7_routing.py`.
Its saved outputs copy was executed independently from outside the repository;
the recount and exported checksums match.

From the repository, rerun with a fresh output directory:

```bash
python3 ../../outputs/phase6_3_validation_routing_20261024/recompute_phase5_7_routing.py \
  --repo . \
  --output-dir /private/tmp/phase5_7_routing_fresh_recount
```

Full suite: 182 tests pass. New tests verify that a changed routing population
stops before creating the output directory, and disagreeing target labels fail
before the count gate. No inference, remote update, or push was performed.
