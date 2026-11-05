# E 5,000-Request Cap and Recovery Seal · 2026-10-28

Latest user instruction "可以直接上5000", continuing the cumulative US$40 cap explicitly confirmed earlier. The original 500 are reused, with at most 4,500 post-pilot calls (including the original 55); this time all 554 complete records are reused, the unresponsive 55th call is skipped but its full US$0.13211 reservation is kept, and it continues from the 56th, with at most 4,445 more calls. Even if all later calls succeed there will be at most 4,999 complete candidates; no re-collection, and no guaranteed SFT count.

`user_acceptance.json` stores the verbatim instruction, the supersession relationship, and the budget; `sample_manifest.json` stores 4,500 post-pilot IDs within Train, keeping a prefix fully consistent with the original 1,500; `execution_receipt.json` stores the inputs, source code, and the import status of the original ledger. The original failure and the 54 responses are in the [timeout archive](../e_teacher_expansion_timeout_20261028/README.md).

The online pre-check checks all tasks and the frozen fingerprints, without opening offline references. API tasks contain only the questions, C1 original answers, original Top-5, and the fixed prompt/schema, with no gold or AI labels. The new code uses the old strict pre-check and does not change the old 2,000 client or the sealed hashes. Afterwards the timeout was raised to 600 seconds and failure stage/duration/HTTP status were added; the model and request body are unchanged, and serial rate limiting and zero retries are kept.

```bash
python scripts/run_e_teacher_5000.py
python scripts/run_e_teacher_5000.py --execute --prompt-key
```

No API by default; actual execution only takes an existing key via hidden terminal input, and the key is not saved. The new run directory is `data/processed/e_distillation_v1/expansion_5000_results_20261028/`. The old and new directories are locked at the same time; imported costs are not double-counted and unknown costs are not released; a newly incurred unknown cost still stops, without automatic skipping. The old 2,000 launcher cannot be used for recovery.

The input compressed archive is physically separated from the offline references; `verification.json` gives the compressed and decompressed fingerprints. Training and student gains have not happened yet; source/length/semantic filtering and the independent Development500 and Selection500 protocols remain in effect. Test11,114 does not enter teacher or training. For the full tests and end-of-work check, see the logs and self_check in this directory.
