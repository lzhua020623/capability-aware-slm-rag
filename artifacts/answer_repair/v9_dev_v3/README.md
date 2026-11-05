# v9 single source namespace interface candidate

Addressing the source identifier ambiguity of v2, the prompt was changed to group by document title and show only citable sentence IDs; it adds clarification of exact IDs, at most five fact citations and lowercase decision. Full Top-5, titles, offset tracing, semantic rules, parser and fallback mechanism, model, samples and budget are retained.

- Frozen input: `answer_repair_v9_dev_v3_gpu.tar.gz`, SHA-256 `4dab82e46dd6b48b0a880c5de38de36418d0c9189c027f12c63ebdfe53e4a940`.
- notebook: `answer_repair_v9_smoke_contract.ipynb`, self-contained private experiment package.
- `implementation_validation.json`: 78 checks, six groups of frozen prechecks; evidence source text, titles and sentence IDs unchanged.
- `engineering_gate.json`: thresholds and stopping rules declared before GPU submission; prompts are not tuned by EM or question semantic performance.

Reproducing the old versions requires using the source code inside their packages. The candidate has not been promoted to default strategy; full development and semantic gain are not yet confirmed. For details see [engineering report](../../../docs/ANSWER_REPAIR_V9_ENGINEERING_20261022.md).

Version 3 (361008203) finished normally at 523.2 seconds, 30 / 30 outputs, six groups of local frozen replay passed. Only 11 / 20 structured outputs were compliant, below the pre-declared 18 / 20 threshold; source ID mismatch errors dropped to zero. Frozen per the stopping rule; these ten questions will not be tuned further, and full development has not been started.

Results are in `kaggle_20261022_v3/`: `outputs.zip` SHA-256 is `9b1a4a117fd018ca89b9bc6f12db0fa50f47c13591c2e3207d8346287bb2572a`; raw results / logs, per-file fingerprints, local replay and the engineering summary are all saved. The three changed answers correspond to two NQ questions; assisted review has been written to `changed_assisted_review.jsonl` and the [review page](kaggle_20261022_v3/changed_review.html), and is not yet counted toward the number of human-confirmed fixes.

`collection_receipt.json` binds the fingerprints of the input package, pre-declared thresholds, raw outputs and derived review files. Replay uses CPython 3.13.16 (GPU was 3.13.15): run `scripts/run_answer_repair_v9.py --replay` from the in-package directory and specify the original `--dataset/--strategy/--all/--c1/--out`. Replay reuses the saved model text and verifies that processing and scoring can be recomputed; it is not regeneration, and does not guarantee GPU outputs are verbatim identical under other software versions; re-inference must save a separate run snapshot.
