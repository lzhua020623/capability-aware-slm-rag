# E 2,000-Candidate Collection Seal · 2026-10-28

Latest user instruction: **2000 first, expand only if valid**. This round is a cumulative 2,000 candidates (original 500 reused + at most 1,500 new), with a cumulative unit-price-estimate stop cap of US$20; the earlier 5,000/US$40 choice has been superseded. Extrapolating from the usage of the original 500, the cumulative total is about US$13.71, of which the new part is about US$10.28; the actual Azure bill has not been verified. New paid calls 0, training 0, student effect not yet measured.

- `budget_acceptance.json`: the latest verbatim instruction, the supersession relationship, the budget, and the boundary of no automatic scale-up.
- `sample_manifest.json`: original 500 IDs + this round's 1,500 IDs, 375 for each of the four AI metadata categories, belonging to Train; not sampled based on gold/EM.
- `execution_receipt.json`: input files, the original 15 shards, source code, and protocol fingerprints. All matched before execution; the original C1/Top-5 are unchanged.
- `teacher_tasks.jsonl.gz`: lossless compressed archive of this round's 1,500 runtime tasks. Contains only the questions, original answers, Top-5, and the fixed prompt/output schema; no offline reference fields.
- `offline_references.jsonl.gz`: physically isolated offline Train references, which do not enter teacher requests; not supervision targets.
- `verification.json`: per-record independent matching against the original C1 shards, purpose/cross checks, compression round-trip, rate-limit package size, and archive fingerprints. Azure was not called.
- `full_tests.log`, `self_check.json`: full regression and end-of-work check (written after the tests complete).

When restoring inputs, only decompress `teacher_tasks.jsonl.gz` to the configured `inputs_directory/teacher_tasks.jsonl`; the online pre-check does not open reference files. Offline review separately decompresses `offline_references.jsonl.gz` to the reference file in the same directory. First compare the compressed/decompressed SHA-256 in `verification.json`, then run:

```bash
python scripts/run_e_teacher_expansion.py
```

Actual collection requires an explicit `--execute`; the key is entered hidden by the user in a local terminal. The client uses an independent process lock and an append-only journal; the original 500 are not resent. A full reservation is made before sending, and settlement uses the fixed unit-price estimate; an unknown cost or version change stops without automatic retry. If paid but the target contract is not met, the record is quarantined and the next registered ID continues, with no re-collection. At most 1,500 new calls; cannot automatically switch to 5,000.

2,000 candidates do not equal 2,000 qualified SFT examples. The original 100 items were released for candidate collection by batch chat approval; this does not create nonexistent per-item web judgments or a 100% semantic accuracy, and the original pilot-production review snapshot is not rewritten. The two assistant disputes keep their temporary training quarantine; targets still need source, 384-token length, and semantic filtering. Only E's Qwen adapter is trained; C1 is unchanged. Whether to continue scaling up depends on the paired net gain on student Development500 after training and on the harmful-repair review; Selection500 gives only one confirmation for the final candidate, and the team Test11,114 stays sealed.

For the related implementation and usage, see the [expansion and cost notes](../../../docs/E_TEACHER_EXPANSION_20261028.md). The historical 5,000-ID inventory is only an offline preparation snapshot and does not constitute API authorization for this round; the original 500 results and the conservative ledger are not overwritten.
