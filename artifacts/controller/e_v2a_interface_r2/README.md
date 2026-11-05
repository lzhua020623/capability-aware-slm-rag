# E v2a r2: Unified Verification Labels · GPU Run Package

For the protocol, see [CONTROLLER_E_V2A_R2.md](../../../docs/CONTROLLER_E_V2A_R2.md); for the sealing record of the acceptance questions, see [fixture_review](fixture_review/README.md).

**Current result: the first GPU run has finished; 10 of 20 items passed, with 1 structural failure, 9 semantic failures, and 0 execution failures. Stopped per preregistration; development calls are 0 and global E is unavailable; the mainline no longer schedules r3.** The [raw results and interpretation](kaggle_v1/README.md) and the [offline deployment decision](deployment_decision.json) are both saved.

## Current Execution Package v2

The execution entry is the [transport_v2 short Notebook](transport_v2/controller_e_v2a_r2_development_dataset.ipynb), with the accompanying `.tar.gz.bin` in the same directory. Source commit `e3ddead`, SHA256: `589dd9750d9e21336e0312609c47ee42d4ae338a86c566da7c15c25cd461ab07`. The only current pointer is in [current_release.json](current_release.json).

v2 aligns to the offline E mode shared by both datasets, and merges in a compatible interface with P60 not enabled. The model, verification prompt, the 18+2 sealed questions, the 384-token single-call budget, and the branch call counts are unchanged. The original v1 never ran on GPU; its original bytes are kept, and it is not used to submit this experiment.

## Historical Unrun Package v1

This package comes from `9d91a40` and was generated before the offline global mode alignment; it never ran on GPU. Its original bytes are kept for audit; before execution the subsequently sealed v2 was used instead; v1 is not treated as the current execution entry.

The source commit is `9d91a40`, and the archive SHA256 is `d4e8d3e6ba0901fd6581e8be05e96e86206ff998e38207b81b79efb98f6ad882`.

| File | Purpose |
|---|---|
| [transport_v1/controller_e_v2a_r2_development_dataset.ipynb](transport_v1/controller_e_v2a_r2_development_dataset.ipynb) | Short Notebook of about 4 KB, submitted to Kaggle to run |
| `transport_v1/controller_e_v2a_r2_development.tar.gz.bin` | Uploaded as a private Dataset; keep the `.bin` suffix to prevent automatic decompression |
| `gpu_release_v1/` | Original archive, embedded Notebook, and execution manifest; sealed only, not submitted directly |

The name differs from r1's `controller_e_v2a_development_v1`. Even if r1's Dataset is mounted by mistake, the Notebook's fingerprint assertion will fail.

## Run Flow

On Kaggle, execute in order:

1. Install dependencies.
2. Run the portable unit tests.
3. Pre-check.
4. Run the 20 acceptance items: all 18 short questions and 2 long questions are run, and failures are classified into three types: execution, structural, and semantic.

Then branch according to the acceptance result:

| Acceptance result | Follow-up run | Process exit code |
|---|---|---|
| All passed | 800 development E calls | 0 |
| Only semantic failures | Both datasets run only Always Repair, 400 calls, in verify_only | 0 |
| Execution or structural failures | Stop; E unavailable | 3 (the Notebook reports an error, but outputs are still exported) |

Whatever the result, `controller_e_v2a_r2_development_outputs.zip` is exported.

## Local Validation

All of the following are CPU checks and **do not equal** GPU acceptance or semantic correctness:

- The full regression tests passed.
- [Real XGrammar 0.2.8 compilation](validation/r2_schema_cpu.json): uses the Qwen tokenizer. The ideal outputs of all 18 questions are accepted; r1's old label `unsupported` and non-insufficient verdicts paired with empty verification IDs are both rejected. The longest short-question prompt is 1,307 tokens; the two long questions are 17,514 and 17,539 tokens respectively.
- [Current v2 unpacking check](validation/extracted_bundle_v2_check.json): 101 file fingerprints match; there are no gold-type fields in the 800 development inputs; after unpacking, the 32 portable tests and the pre-check all passed. The original v1 check is kept as a historical record.

## Independent Review

After obtaining the outputs, no model call is needed and gold is not read:

```bash
python scripts/audit_controller_e_v2a_outputs.py --archive <outputs.zip> --release-dir artifacts/controller/e_v2a_interface_r2/gpu_release_v2
```

The audit script recomputes each item's failure category and acceptance result from the raw records, and checks that the development outputs fall strictly within the corresponding branch: the semantic-failure branch must not replace any answer.

## Executed Stop Branch

- The GPU's 18 short + 2 long questions are complete; the result is failed, and the number of development predictions is 0.
- Per the original registration, the global E mode has been frozen offline as unavailable; the 400/800 development runs are not executed, and r3 is not started.
- The real development compliance rate was not measured and is not filled in as 0 or 100%; the fictional-question contract ratio does not substitute for this statistic. The enabling gates registered before the run are kept unchanged.

## First GPU Launch · 2026-10-26

Confirmed Running, Private, GPU T4×2 on [Kaggle Version 1 / 362285592](https://www.kaggle.com/code/yearo111/controller-e-r2-shared-frozen-20261026?scriptVersionId=362285592). The package's 32 tests and the 800-input/20-acceptance pre-check passed. The first launch observation is in [launch_v1.json](launch_v1.json); this record does not mean the real semantic or long-input gates passed; later results are saved separately, without overwriting this record.
