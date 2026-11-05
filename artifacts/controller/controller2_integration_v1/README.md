# Controller 2 Integration Validation · 2026-10-27

For sources and scope, see the [handover](../../../docs/CONTROLLER2_INTEGRATION.md) and `merge_sources.json`.

- **Full regression: all 1,224 tests passed, 104.634 seconds**, `full_tests.log`.
- The retrieval-enhancement-specific 50 tests, reranking-and-integration-specific 329 tests, and historical-sealed-specific 10 tests passed; the specific suites overlap and are not counted as independent samples or test counts.
- 6 new merge-boundary regression tests cover type migration, traces for rejected candidates, same-ID text truncation, FEVER support, E unavailable, and default model configuration.
- The old T5 wrapper regression was changed to restore the original sealed source code; a new check confirms that the new Controller source code is rejected by the old seal. `full_tests_initial_frozen_guard.log` keeps the record of the regression that was first blocked by the strict hash check.
- The pre-fix logs for the historical Windows package-test source setup and for missing NLTK/pandas are also kept; production source hash verification was not relaxed.
- `before.json` is the start-of-work state, `self_check.json` is the end-of-work check, and `source_snapshot.json` pins the source tree actually tested; `checksums.json` verifies the files in this directory.
- `environment.json` records the CPU test environment, NLTK 3.9.2, and the SHA-256 of the four resource files of English punkt_tab; NLTK data and model caches are not committed.

Both source branches are in the new branch's history. The original `controller` branch, old frozen configs, packages, historical results, AGENTS.md, and uncommitted user review files are all kept. Source report bytes are identical to upstream; new reranking weights/truncation/answer selectors were not enabled, no teacher data was generated, and no training/GPU/API calls were started.

The production pre-check still exits with 2 as expected: the original P60 online feature code and the training/tuning IDs of the three models have not been delivered; E is still unavailable for both datasets. `readiness_after.json` does not mean GPU engineering or effectiveness acceptance has passed.

The next steps are team partitioning and per-question baseline integration, cross-route training ID isolation, completing the P60 materials, and E teacher qualification validation. The current integration is not presented as a claim of accuracy improvement.
