# v8r2: 8-item real-model engineering acceptance

Only generation stopping, numbering and citation display format are fixed; the three-stage semantic decisions and the v7 control keep their original behaviour. For rules and pass conditions see [v8r2 notes](../../../docs/ANSWER_REPAIR_V8_R2.md). The original 75-item failed experiment is fully retained.

- Fixed N1/N2/N8/N12/N15/N31/N48/N96: 4 correct controls, 4 E targets, all from existing development samples.
- 8 items per group across four groups, 32 outputs in total, at most 64 generations; the same frozen Qwen7B NF4 setup.
- Input package SHA-256 `ca318d513f5feab0ff2f4c94acf19888e46ef0021ec454ef6dfd43f166bf58fe`, all 93 payload files verified.
- `validation.json`: 106 portable checks passed, and 32 simulated pipeline records passed CPU replay; simulated results are not GPU results.
- `manifest.json` and `preparation.json`: pre-run frozen snapshots, not changed to a post-hoc acceptance status.
- `cloud_status.json`: records the Kaggle status separately; version 1 ran successfully for 585.8 seconds, and all 32 real predictions completed frozen replay; the original engineering threshold failed, and the full pipeline had no changed answers. See [results report](../../../docs/ANSWER_REPAIR_RESULTS_20261019_V8_R2.md).
- Notebook: `notebooks/answer_repair_v8r2_smoke8.ipynb`, compatible with Kaggle auto-extracted input; the actual inputs, original answers and gold all come from the original frozen package, and no runtime reference answers or labels were injected.

Engineering pass requires meeting the pre-fixed conditions of format validity, three-stage coverage, traceability to the source text, untruncated raw generation and call budget. Only after passing can the full 75-item development experiment be rerun; it does not represent semantic gain or strategy promotion.
