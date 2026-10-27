# Latest reranking integration · 2026-10-27

Upstream `1d518562daa97ad555657a1965db4b8e02815c69`, three commits after `5583285`. See [integration notes](../../../docs/RERANK_1D51856_INTEGRATION.md).

94 targeted tests passed. **1305 tests passed in 173.332 seconds** against `a936187c564b2fbd3bc8547e63fc0e2965fddc82`. See [full log](full_tests.log), [post-work check](post_selfcheck.json), [source fingerprints](source_snapshot.json) and [file fingerprints](checksums.json). New deeper retrieval is opt-in, uses registered resource preflight, and is not enabled in the production Controller. New selector weights are not promoted; training usage IDs remain unverified. No teacher calls, training or real GPU benchmark was run. The existing 500-task E pilot is unchanged.

The integration preserves upstream history and aggregate results. [Integration receipt](integration.json) records the two failed local test attempts and the corrected isolated dependency injection; these were engineering checks, not model experiments.
