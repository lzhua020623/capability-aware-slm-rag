# Final round v8r4 FACT verification smoke

Only the verification prompt and its ten-line contract are changed; minimal repair, candidate source checks, model and the 384 token budget are not changed. Independent package SHA-256: `d13880c5e114b5f8379f931103af249e1f10b4975de98c9a5b74a9c161900e11`.

- 158 local checks and 118 frozen portable checks passed; 103 payload files verified one by one, and the Notebook compiles.
- 32 simulated pipeline records completed frozen replay; they are not model experiment results and are not included in research results.
- Fixed original 8 items, 32 real outputs in four groups, upper limit 64 generations; for the actual status see `cloud_status.json`.
- The r3 engineering threshold is reused, plus three full-pipeline verification regressions N31/N48/N12; EM/F1 do not select the version.
- If all pass select r4, otherwise keep r3; no r5 is added on these eight items. The subsequent citation failure of N31 is still recorded, and the source constraint is not forcibly relaxed.
- `manifest.json` and `preparation.json` are pre-run snapshots and cannot be changed to a passed status afterwards.

For the plan see [r4 notes](../../../docs/ANSWER_REPAIR_V8_R4.md), and for the wrap-up plan see [final experiment protocol](../../../docs/ANSWER_REPAIR_FINAL_PROTOCOL.md). New outputs and human conclusions are still pending the actual run and review.
