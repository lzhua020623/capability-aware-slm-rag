# v9 first engineering snapshot (failed, fully retained)

- Input: `answer_repair_v9_dev_v1_gpu.tar.gz`, SHA-256 `4f3e3e2acf4de62bf70089b2a5ee13b66afc917641c8f602f6dc6faaa79046bf`.
- Kaggle Version 1: `360999513`, Private, T4×2, CUDA OOM after 592.9 seconds.
- Output ZIP: `kaggle_20261022_v1/outputs.zip`, SHA-256 `389dce00a3efb40effe4ee28106bf17bf32a7f78a52bc219dd9c0ef0ef7bd91f`.
- `archive_manifest.json` records all 92 ZIP members. The full source and input copies are saved in the original ZIP and the frozen input package; the working extraction directory is not committed again file by file.
- The original `results/`, `environment.json`, `bootstrap.json` and run logs are retained; `local_replay.json` records five groups of full replay and one group of prefix replay; `engineering_summary.json` records 26 / 30 saved results and invalid outputs.

The first engineering threshold was not passed, and no gain is claimed. Format compression is set up separately as `v9_dev_v2`, without changing the bytes of this package. For the full description see [engineering report](../../../docs/ANSWER_REPAIR_V9_ENGINEERING_20261022.md).
