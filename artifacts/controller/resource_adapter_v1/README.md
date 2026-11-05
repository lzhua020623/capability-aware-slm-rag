# Controller resource interface acceptance · 2026-10-24

Delivered this round: NQ full offset location validation, explicit resource fingerprint registration, lazy Controller provider, real retrieval replay of development queries; FEVER original package import, full ID/body-text order check and FAISS search read-back acceptance.

`validation.json` records the fingerprints of new/changed source code and tests, the actual CPU environment, test counts and each validation entry point. `tests.log` is the full regression, **all 706 tests passed**; `targeted_tests.log` is a 31-test targeted regression of NQ registration/provider and the existing Controller interfaces.

`protocol_check.json` verifies the original T1 design hash and the 68 reference Git blobs; it reports only two explicit source differences, in runner/providers, and does not modify the frozen design. `data_check.json` checks the complete T2 freeze and the 552 source fingerprints. The resource-gap description in the old T2 report is a snapshot from freeze time; this round updates progress by appending acceptance records, without rewriting historical data.

This round does not modify the original prompts, classifier, thresholds, budgets, Top20/Top5 or sample splits. The new NQ `--nq-resources` is mutually exclusive with the old metadata path, and the old interface is retained. The new read layer provides only passage_id/text; retrieval results additionally include the original scores, and offline annotation fields do not enter the runtime.

`local_mmap_search_diagnostic.log` / `local_heap_search_diagnostic.log` are local search diagnostics from when the CPU replay took a long time: both loading methods actually completed the original full-store search, with no crash. This is a single diagnostic under parallel load and cannot serve as a controlled speed comparison; formal costs must be re-measured in the locked GPU execution environment.

**The formal execution seal is still not complete**: the FEVER provider, the fixed 900-item new FEVER C1, the Full/No Probe/human-type comparisons, strict fully paired statistical acceptance and CUDA environment locking continue to proceed under T1/T3. CPU and resource checks do not constitute proof of accuracy gains.
