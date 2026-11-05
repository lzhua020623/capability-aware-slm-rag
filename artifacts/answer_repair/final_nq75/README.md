# E final 75-item development experiment

The final inference version is r3. Of the final three pre-declared r4 regressions only N12 passed, so per the original rule it fell back to r3; the version was not selected based on EM, and no r5 is added.

The original 75 items are 11 confirmed internal E targets and 64 correct controls. C1 is reused, and four new groups of 75 items each are added: v7 / full r3 / without original check / without candidate check; 300 outputs in total, with an upper limit of 600 generations. Used only for the final development description; no further tuning after completion, and not treated as evidence of unseen gains.

- Input archive: `answer_repair_final_r3_nq75_gpu.tar.gz`.
- SHA-256: `b5be903db6efecb5681498f0110ebed65920369ca1e5d466492ebae7455fc98b`.
- Inference source: 29 files match the selected frozen r3 source one by one; only non-inference files such as the evaluation-list recognition and packaging scripts were adapted as recorded.
- Tests: Python 3.13 passes the original 111 portable checks of this frozen version. The AST compatibility adapter runs only in the unittest process and does not modify the frozen package or the model process.
- Notebook: `notebooks/answer_repair_final_nq75.ipynb`.
- Private input: <https://www.kaggle.com/datasets/yearo111/e-final-nq75-frozen-r3-20261020>.
- Run version: <https://www.kaggle.com/code/yearo111/e-final-nq75-r3-comparison-20261020?scriptVersionId=360070600>, T4 ×2.

`manifest.json` and `preparation.json` are frozen preparation snapshots, and `validation.json` is the local validation; for the actual cloud status see `cloud_status.json`. On 2026-10-21 the 29 raw exported files were collected and passed the 300-item frozen replay. v7 EM 20→23, full r3 20→20 (changed only one question, N96), and the two ablations were 19 and 21 respectively. Assisted opinions have been completed for the 15 changed answers of the main strategy. These 15 items have been included in F1–F15, and the 42-item full list was fully processed before 2026-10-22; for the conclusions see data/annotations/human/answer_repair_final_v1_batch04_20261022.summary.json. The output ZIP fingerprint and per-file SHA are in `result_validation.json`, the unified list is `../final_review_v1/F42_review.html`, and the full explanation is in `docs/ANSWER_REPAIR_RESULTS_20261021_FINAL.md`.
