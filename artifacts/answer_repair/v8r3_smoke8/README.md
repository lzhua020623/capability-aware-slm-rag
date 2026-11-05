# v8r3: independent engineering acceptance of the candidate format fix

Fixed 8 development samples, 32 predictions in four groups, at most 64 generations. Only three explicit candidate grammars are made compatible, and format, source and length rejections are distinguished; semantic prompts, model, evidence, references and budget are all unchanged. The hashes of the 33 actual r2 stage prompts all match r3.

- [Plan and pre-set thresholds](../../../docs/ANSWER_REPAIR_V8_R3.md)
- `manifest.json` and `preparation.json` are pre-run frozen snapshots and are not rewritten after the run.
- `validation.json`: 151 local checks and 111 frozen portable checks passed; 98 payload files verified one by one, the Notebook compiles, and CPU replay of 32 simulated pipeline records passed. Simulated results are not stored as model results.
- `offline_candidate_diagnostic.json`: only uses the new parser to check the 33 stage raw texts of the old r2; 6 field omissions are recoverable, and 3 source rejections are still kept. No new model calls, no inference of new final answers, and the r2 failure record is not changed.
- Frozen package SHA-256: `72ed761fd8017ebbaf439729ff318caa55c0823acf4b62429c4db8f31310b6f1`.
- `cloud_status.json` records the actual cloud status separately; Kaggle version 1 (359908613) ran successfully for 553.1 seconds, T4 ×2; 32 outputs, 43 generations, frozen replay and the new engineering threshold all passed. See [actual results](../../../docs/ANSWER_REPAIR_RESULTS_20261019_V8_R3.md).

The new threshold requires zero format errors, and reports source or length rejections separately. Source rejections are not counted as successful repairs. The r2 original threshold failure record remains unchanged. An engineering pass also does not represent semantic improvement, and the new answers still need review; the full 75-item rerun will not be started automatically.

The full pipeline only fixed N96, with the assisted suggestion being the correct naming origin; EM 1→1. N8/N31 are still wrongly accepted, and the correct candidate of N48 is still wrongly rejected. Assisted opinions for the 10 changed-answer records have been saved, the human review fields are pending, and a reliable net improvement cannot be claimed.
