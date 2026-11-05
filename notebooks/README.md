# E experiment Notebooks

This directory keeps only the latest completed v7 Notebooks. v7 has been run, and the held-out set is no longer used to select new policies; the next round, v8, has not been implemented yet. Existing results can be replayed on CPU, without calling the GPU again.

| Experiment | Colab | Kaggle |
|---|---|---|
| v7 main package, 541 results | [Main package](answer_repair_colab_v7_reviewed.ipynb) | [Main package](answer_repair_kaggle_v7_reviewed.ipynb) |
| v7 batch 4, 300 NQ3216 stratified held-out samples | [Batch 4](answer_repair_colab_v7_heldout3216.ipynb) | [Batch 4](answer_repair_kaggle_v7_heldout3216.ipynb) |

For how to run, see [Colab and Kaggle instructions](../docs/ANSWER_REPAIR_COLAB.md); for conclusions see [v7 results](../docs/ANSWER_REPAIR_RESULTS_20261014_V7.md).

The 8 Notebooks from v1 to v6r1 have been moved to `history/answer_repair/`; the entry point is the [History index](../docs/history/answer_repair/README.md). The old Notebook contents were not modified, and the original input packages and results are still in `artifacts/answer_repair/`.
