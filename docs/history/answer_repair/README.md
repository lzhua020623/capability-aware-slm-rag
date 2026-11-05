# E historical archive

This directory stores the documents, experiment designs, and early progress of v1 through v6r1. The current working base is v7; the next round, v8, is not yet implemented. For current notes, see [project progress](../../PROJECT_STATUS_AND_DIRECTION.md), [E handoff](../../ANSWER_REPAIR_HANDOFF.md), and [v7 results](../../ANSWER_REPAIR_RESULTS_20261014_V7.md).

## Document index

| File | Purpose |
|---|---|
| [ANSWER_REPAIR_RESULTS_20261013.md](ANSWER_REPAIR_RESULTS_20261013.md) | Real results of v1 through v4, the Reanswer comparison, and Controller integration testing |
| [ANSWER_REPAIR_RESULTS_20261014.md](ANSWER_REPAIR_RESULTS_20261014.md) | v5 and v6r1 results, the 128-record assisted review, and key human confirmations |
| [ANSWER_REPAIR_REVIEW_20261013.md](ANSWER_REPAIR_REVIEW_20261013.md) | Selection, assisted review, and sources of the 37 development candidates |
| [ANSWER_REPAIR_V5.md](ANSWER_REPAIR_V5.md) | Frozen protocol for the reviewed development set and same-model second verification |
| [ANSWER_REPAIR_V6.md](ANSWER_REPAIR_V6.md) | Frozen protocol for NQ verbatim extraction and FEVER independent MNLI |
| [STRATEGY_HISTORY.md](STRATEGY_HISTORY.md) | Early JSON format, citation matching, candidate generation, and strategy evolution |
| [GPU_RUNS_V1_V6R1.md](GPU_RUNS_V1_V6R1.md) | Historical Colab and Kaggle execution, package hashes, and Controller integration testing notes |
| [PROGRESS_UPDATES_20261013_14.md](PROGRESS_UPDATES_20261013_14.md) | Stage progress, routing trial runs, conclusions at the time, and data limitations |
| [HANDOFF_20261014.md](HANDOFF_20261014.md) | Handoff snapshot before reorganization, keeping old check counts, interface examples, and later supplements |
| [archive_index.json](archive_index.json) | Old paths, new paths, and before/after SHA-256 of this migration |

The human confirmation criteria remain at the original path [ANSWER_REPAIR_CONFIRMATION_20261014.md](../../ANSWER_REPAIR_CONFIRMATION_20261014.md), to keep references to human source records stable. Human records and gold are not moved into the history directory.

## Notebook index

Historical notebooks are located in [notebooks/history/answer_repair](../../../notebooks/history/answer_repair/). The move only changed the directory; their contents are exactly the same as before the move.

| Version | Notebook |
|---|---|
| v1 | [Colab](../../../notebooks/history/answer_repair/answer_repair_colab.ipynb) |
| v2 | [Colab](../../../notebooks/history/answer_repair/answer_repair_colab_v2.ipynb) |
| v3 | [Colab](../../../notebooks/history/answer_repair/answer_repair_colab_v3.ipynb) |
| v4 | [Colab](../../../notebooks/history/answer_repair/answer_repair_colab_v4.ipynb) |
| v5 | [Colab](../../../notebooks/history/answer_repair/answer_repair_colab_v5_reviewed.ipynb) |
| v6 | [Colab](../../../notebooks/history/answer_repair/answer_repair_colab_v6_reviewed.ipynb) |
| v6r1 | [Colab](../../../notebooks/history/answer_repair/answer_repair_colab_v6r1_reviewed.ipynb) / [Kaggle](../../../notebooks/history/answer_repair/answer_repair_kaggle_v6r1_reviewed.ipynb) |

Each notebook loads its own frozen package; swapping the input package cannot be used to pass off a new strategy. The historical build scripts are still kept, and their default output now points to this directory; existing experiment packages and notebooks do not need to be regenerated.

## Original experiment materials

Input packages, downloaded ZIPs, per-sample predictions, run logs, and review materials remain under `artifacts/answer_repair/` at the repository root; the frozen bytes were not moved or modified:

- `v1/`, `v2/`, `v3/`: early input packages and raw exports.
- `session_20261013/`: combined raw export of v4 and Controller integration testing.
- `v5_reviewed/`, `v6_reviewed/`, `v6r1_reviewed/`: reviewed development experiments and the corresponding frozen materials.
- `review_20261013/`, `handoff_20261014/`: early review sources and delivery snapshots.

See [git_archive_manifest.json](../../../artifacts/answer_repair/git_archive_manifest.json) for exact paths and hashes. Reproduction uses the frozen code in the input packages; path adjustments in document links and current build scripts do not rewrite old packages.

## Early Reference RAG engineering validation

All of the following are 20-sample smoke runs that validate engineering connectivity and cannot replace the 500-sample formal pre-experiment; FEVER uses a reduced controlled candidate pool.

| Historical trial run | Metric | Retrieval and errors |
|---|---|---|
| NQ 3B v2 | EM 0.3000, F1 0.3460 | gold in Top-5: 17/20; retrieval miss 3/20; hit but answered wrong 11/20 |
| NQ 7B | EM 0.4000, F1 0.4770 | gold in Top-5: 17/20; retrieval miss 3/20; hit but answered wrong 9/20 |
| FEVER 3B | accuracy 0.8000 | gold in Top-5: 20/20; hit but answered wrong 4/20; not a formal retrieval result on the full corpus |

Raw outputs are still in `results/base_rag/`. These historical model sizes differ from the current frozen Qwen 7B configuration and should be interpreted under their respective experimental criteria.
