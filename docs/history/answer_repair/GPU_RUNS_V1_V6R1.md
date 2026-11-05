# E historical GPU execution record

This is a historical note extracted from `docs/ANSWER_REPAIR_COLAB.md` during the 2026-10-16 reorganization, keeping the run status and conclusions of that time. For current status, refer to [project progress](../../PROJECT_STATUS_AND_DIRECTION.md), [v7 results](../../ANSWER_REPAIR_RESULTS_20261014_V7.md), and [current handoff](../../ANSWER_REPAIR_HANDOFF.md).

2026-10-14: the 112 real predictions of v5 and the 128 of v6r1 have all been archived and passed frozen-code replay; see the [results report](ANSWER_REPAIR_RESULTS_20261014.md). The latest reproducible version is [Colab v6r1](../../../notebooks/history/answer_repair/answer_repair_colab_v6r1_reviewed.ipynb) or [Kaggle v6r1](../../../notebooks/history/answer_repair/answer_repair_kaggle_v6r1_reviewed.ipynb), with `artifacts/answer_repair/v6r1_reviewed/answer_repair_v6r1_reviewed_gpu.tar.gz`. See [v6](ANSWER_REPAIR_V6.md) for the run protocol. After the Colab GPU quota was exhausted, all cells were completed in a [private Kaggle draft](https://www.kaggle.com/code/yearo111/e-answer-repair-v6r1-reviewed/edit); verification of 48 files, 81 cloud tests, and inference for all four groups passed. The 30 output files have been downloaded and the GPU session released.

Latest results directory: `artifacts/answer_repair/v6r1_reviewed/kaggle_20261014/`; ZIP SHA-256 `3e9cd5eec2a658cf9ff5146c2b7322c9166d35367b9af34696439847589df684`. The human review queue is `v6r1_reviewed/release_audit/review_priority.md` (reading) and `review_priority.jsonl` (editable); human judgements are not complete in either. There is no need to re-run inference for existing results.

v6r1 archive SHA-256: `74d40b01c96245e1caf2ac8af786bf3b05394971863bddc81a75a2a2afa45ef5`. The notebook contains 7 code cells, which in order perform verification, dependencies and 81 tests, the 28 reviewed development samples, the original dev50 × 2 regression, and result export. New and old packages each have their own independent output directory. The FEVER/NQ strategies each apply only to their corresponding dataset, and classifier cost is reported separately. Do not re-run all historical cells of the existing smoke notebook.

Kaggle automatically extracts uploaded `.tar.gz` files. The dedicated notebook supports both the original archive and an extracted directory: for the latter, it first checks the fixed SHA-256 `1c095e9fb52077898e8eb85672f76ba4ac052e87b9e099da61df523be404d0bd` of `bundle_checksums.json`, then verifies all 48 files and saves `input_integrity.json`. Pre-extracted mode does not claim to re-verify the bytes of the original archive. This adaptation only changes file loading and does not change the frozen experiment code; both loading paths and rejection of tampered files/checksum manifests have been verified locally. The validation record is at `artifacts/answer_repair/v6r1_reviewed/kaggle_transport_validation.json`.

After exporting results, replay locally using only the code inside the frozen package:

```sh
python3.11 scripts/audit_answer_repair_release.py --bundle-dir artifacts/answer_repair/v6r1_reviewed/gpu_bundle --results-dir artifacts/answer_repair/v6r1_reviewed/kaggle_20261014 --out-dir artifacts/answer_repair/v6r1_reviewed/release_audit
```

The following is the v4 historical execution record.

The v4 portable package is `artifacts/answer_repair/v4/answer_repair_v4_gpu.tar.gz`, corresponding to the [v4 notebook](../../../notebooks/history/answer_repair/answer_repair_colab_v4.ipynb). The real experiment was executed on [Colab T4](https://colab.research.google.com/drive/1XI0qoiW0NIQHiD7iOX8ewqM07Y-snxMQ); local code has not yet been committed or pushed.

v4 package SHA-256: `2a40d4fbaff591b02d116243f92404731cb71ccc5ce8a77092f4fda70dd3156e`. Packages and outputs of historical v1/v2/v3 are kept separately; different code versions use independent output files.

1. In Colab, open the v4 notebook via "File → Upload notebook" and choose a T4 GPU. Kaggle can import the notebook; enable GPU and Internet and add the v4 package as a private input.
2. Run the CUDA check and upload the v4 package. The notebook verifies the archive hash and the hashes of each code/input file.
3. Run dependency installation, environment checks, 54 CPU tests, and fixed-sample pre-checks. The first real call downloads Qwen2.5-7B-Instruct; subsequent calls reuse the cloud cache.
4. Compare `verify_repair` and `grounded_reanswer` on 10 samples each of FEVER/NQ, 40 extra generations in total, and review all outputs.
5. The optional development cell defaults to `RUN_DEVELOPMENT=False`. After review, the selected E strategy and the `reanswer` comparison can be run, 50 samples per dataset, 200 generations in total. The candidate strategy is explicitly selected in `SELECTED_E_STRATEGY`.
6. The last cell exports a ZIP of predictions, summaries, environment, input manifest, file checksums, and logs. The human columns in the review template are empty; generating the template does not mean review is complete.

Only per-sample NF4 inference is done; no retrieval index rebuild is needed. Top-5 content/order is fixed, with at most one call per E sample and a 384-token cap. Reference answers participate only in offline scoring. Neither NQ literal citation matching nor answer span checking amounts to semantic support verification.

The output directory is `/content/answer_repair_v4_outputs` on Colab or `/kaggle/working/answer_repair_v4_outputs` on Kaggle. Drive is not mounted automatically. Runs can be resumed within the same runtime; after the runtime is released, the package must be re-uploaded and the old outputs restored. Export results before release.

## Completed engineering smoke

Real T4 v4 results on 10 samples each on 2026-10-13:

| Dataset | C1 correct | Verify correct / invalid | Independent candidate correct / invalid |
|---|---:|---:|---:|
| FEVER | 7/10 | 7/10, 2 invalid | 8/10, 0 invalid |
| NQ EM | 2/10 | 2/10, 4 invalid | 2/10, 4 invalid |

The one FEVER label that was fixed cited Venezuela's Valencia Municipality, which cannot prove the population proposition about the target entity and its administrative centre; it is not counted as a reliable grounded repair success. The NQ independent candidate changed the text of two samples, but EM had no gain, and the tibia/fibula sample still answered the relative position of a single bone. Keeps that passed the citation check also include cases citing unrelated entities.

v4 fixed the issue of a trailing backtick attached to the last token of complete JSON: the raw output and the object text actually submitted for parsing are saved, and the strict schema and quote checks remain unchanged. The fixed 50-sample development results are complete: FEVER 45→46, but the new correct one is the entity-mismatched Valencia; NQ 20→20, with 1 fixed and 1 broken. Reanswer had no gain in either set. Smoke is a subset of that development set and is not merged into the statistics as an independent second batch. See the [results report](ANSWER_REPAIR_RESULTS_20261013.md) for full outputs, cost, cases, and package hashes.

## Regenerate the notebook

```bash
python3.11 scripts/build_answer_repair_notebook.py \
  --bundle artifacts/answer_repair/v4/answer_repair_v4_gpu.tar.gz \
  --out notebooks/history/answer_repair/answer_repair_colab_v4.ipynb \
  --strategy grounded_reanswer
```

The generator embeds the archive hash and syntax-checks the 8 code cells. An existing archive refuses to be overwritten by different code; the next algorithm version uses a new package name and directory; documentation inside old packages is likewise a code snapshot of that time.

## Controller and E integration

E's execution interface for the final `ControllerDecision` is described in [ANSWER_REPAIR.md](../../ANSWER_REPAIR.md). The upstream branch is frozen at `7c7863f98f87e892119481a79e23ca6dcebdced0`. The isolated integration package `artifacts/answer_repair/e_controller_gpu.tar.gz` reuses that branch's Diagnosis, Probe, Safe Binary, Controller, and v4 E; no retraining or threshold tuning.

It only validates Controller→E: the other two types of enhancement actions are only recorded, not executed. Probe and E share one NF4 model, with calls/tokens/generation latency recorded separately. The fixed development samples come from the training intersection, so this result cannot be treated as held-out or as the effect of the complete adaptive system.

The actual integration has run the fixed 100 samples on T4: 27 Probe calls, 1 E call; neither FEVER nor NQ correct counts changed. Code, raw outputs, and run logs have been exported and replay-verified record by record. The combined download package is [E_answer_repair_20261013_results.zip](../../../artifacts/answer_repair/session_20261013/E_answer_repair_20261013_results.zip).
