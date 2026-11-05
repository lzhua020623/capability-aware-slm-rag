# T2 Validation Record · 2026-10-23

- `tests.log`: full CPU regression, 665 tests passed; 17 were added in this run.
- `data_validation.json`: checks of the frozen samples, references, held-out pool, and source file hashes; does not compute EM/F1 or new gains.
- `readiness_check.json`: `--require-full-t2` returns 2 as expected; the FEVER new C1 and matching index resources are still pending sealing.
- `branch_resource_inventory.json`: check of the C1/resource file trees of 11 remote branches, including controller; `.index`/`.npy` are not tracked. Resources ignored by Git or archive internals are not inferred to be nonexistent.
- `validation.json`: test and check commands, results, and artifact hashes.

This is not a GPU experiment result and cannot be used to report QA improvement, a fully clean knowledge base, or that the new FEVER C1 is complete.
