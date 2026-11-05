# FEVER Provider Acceptance

`query_replay.json/log`: the fixed first 10 exposed historical C1 queries, the real fixed BGE snapshot, the original IVFPQ(nprobe=16), and lazy offset reading; for 10/10 the Top5 IDs and order match the original C1, and the first five of Top20 also match. Historical predictions, gold, or new final questions were not used for judgment.

`tests.log`: 726 CPU regression tests across the whole repository passed. Targeted tests cover FEVER prompt/label fallback/one-retrieval re-answer, index and dataset dispatch, the resource hash chain, coverage of all 900 C1 fictional engineering samples, source/input drift, and interruption saving. `initial_targeted_tests.log` keeps one failure from the old smoke expectation that FEVER is unsupported; the test was updated to explicitly verify the new FEVER search and re-answer, not skipped.

`protocol_check.json`: the original T1 design and the 68 reference Git blobs are complete; it registers the current source diffs of P3/execution/providers/runner and related tests, without writing back to the frozen protocol. `data_check.json`: the T2 splits and all 552 source fingerprints are unchanged.

The full execution status and source/file fingerprints are in `validation.json`. CPU validation does not mean real Qwen GPU inference has been completed, let alone that QA metrics improved.
