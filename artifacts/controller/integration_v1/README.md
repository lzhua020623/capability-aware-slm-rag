# Controller integration v1

This directory stores the provenance and engineering validation of the first `controller` integration round on 2026-10-23; it is not a new benchmark result.

- `branch_manifest.json`: all 10 remote source commits have been merged; Git blob comparison of 764 unique historical data/result/model/split paths.
- `initial_merge_tests.log`: raw record of the first post-merge round of 606 tests, with 8 compatibility failures.
- `tests.log`: complete record of all 635 tests passing after the fixes and the added integration/scoring tests.
- `cpu_test_environment.json`: the actual CPU regression environment; not an acceptance of the CUDA experiment environment.
- `smoke/`: execution traces and manifests for two fictional inputs × five modes; fake provider outputs, not counted as NQ/FEVER quality results.

For methods, reproduction commands and limitations, see [Integration handoff](../../../docs/CONTROLLER_INTEGRATION.md). Historical E human records, predictions and data splits were not rewritten.
