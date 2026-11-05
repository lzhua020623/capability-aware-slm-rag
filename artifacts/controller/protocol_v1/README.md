# T1 protocol freeze validation · 2026-10-23

Protocol: [CONTROLLER_EXPERIMENT_PROTOCOL_V1.md](../../../docs/CONTROLLER_EXPERIMENT_PROTOCOL_V1.md). Reference implementation: `86a7b126225f7cbc5ffe4c7af7db3137acbec9f5`.

- `design_audit.json`: checks of the protocol, documents, reference snapshot and Git blobs; 68 reference files passed, with no drift from this workspace.
- `execution_readiness.json`: formal execution still has T2/T3 thresholds. `--require-execution-ready` returns 2 as expected; this is not a test failure.
- `checker_validation.json`: exit codes of the two check commands and the scope of this run.
- `tests.log`: full CPU unittest regression, 648 tests passed, of which 13 are new protocol tests. Covers budget calculation, missing comparison groups, source/prompt drift, configuration inconsistency, and not falsely claiming execution readiness.
- `validation.json`: test commands, counts, CPU dependency versions and artifact hashes.

This run did not read any new benchmark questions, run GPU inference, train models or produce new QA gain numbers. The existing runtime configuration and algorithm source code keep their original bytes. The CPU test environment differs from the planned deployment environment in `requirements-controller.txt`; this result does not prove that the CUDA dependencies or model loading have passed acceptance.

T3 will create an independent execution manifest that references this protocol and registers new capabilities and implementation differences; it must not overwrite the historical records here.
