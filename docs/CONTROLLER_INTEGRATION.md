# Controller integration handover · 2026-10-23

2026-10-27: the current main line is [the 24 October Diagnosis/P60/P64](CONTROLLER_1024_BASELINE.md), with the production entry point `src.pipeline.current.load_current_pipeline()`. The text below keeps the original integration executor and historical reproduction notes; the old commands do not mean P60 is already enabled.

2026-10-26: for the current stage and the newly added merge tasks see the [T1–T9 task list](CONTROLLER_TASKS.md). Diagnosis's `138d52a` will be integrated in layers, keeping the existing default metered executor and Probe; P60's training/tuning exposure, online features, cost and effect are pending independent acceptance and are not the current default routing. E r2's [registration](CONTROLLER_E_V2A_R2.md) and global mode are aligned before GPU; neither the selection run nor the final evaluation has been started.

2026-10-25 latest entry point: [T4 full results and interpretation](../artifacts/controller/t4_v1/README.md). T3 real-CUDA engineering acceptance and local acceptance have passed; T4's 2,800 paired predictions have been accepted, automatically scored and replayed consistently from an independent ZIP, and the full 774-item regression passes. The Kaggle failure was a summary KeyError after all predictions were saved; the original manifest is kept, and the current summary interface has been patched. Adaptive: NQ EM tied at 78/200, FEVER 170→174/200; human semantic/warning/trace acceptance is not complete, and T5 or final evaluation has not been carried out.

Subsequently, with user authorisation, work proceeds per [E v2a and the single NQ routing amendment](CONTROLLER_E_V2A_AND_T5_V1.md): first register and get the single constrained verification working end to end, with the development set determining E's run mode; the single optimisation candidate is to add retrieval enhancement for low-confidence kept NQ questions. Self-checks at the start and end of every round, commit after testing. The original T1/T3 and T4 results are unchanged; before the selection run, the user still needs to confirm the 15 key NQ routing cases.

## Early integration and acceptance records (2026-10-23—24)

The currently active development branch is `controller`. This round merged that day's 10 remote branches and implemented a unified execution and offline evaluation entry point from frozen C1 outputs to final answers. This version is for establishing the system baseline; no E strategy has been promoted to a proven-effective default; real GPU end-to-end quality experiments have not been completed, and Qwen training or RSI has not started.

The subsequent experiment design is frozen in the [T1 unified experiment protocol v1](CONTROLLER_EXPERIMENT_PROTOCOL_V1.md): six main comparison groups, two explanatory controls, per-group budgets, paired scoring and a limit of one round of focused optimisation. T2 samples are frozen and resource acceptance is still to be completed; T3 still needs to complete Full, FEVER retrieval and the controls/acceptance; the "exists" descriptions on this page correspond only to the reference implementation and do not mean all protocol groups are executable.

T1's full CPU regression is 648 passed (13 new protocol tests), with no drift across 68 reference files; the verification archive is at [protocol_v1](../artifacts/controller/protocol_v1/README.md). The 635 items in Section 4 below are the historical acceptance count of the earlier integration commit.

T2 has frozen [the samples and complete evaluation held-out pools of both datasets](CONTROLLER_DATA_V1.md): NQ 7,564 and FEVER 1,886 items are forbidden for training/tuning; 500 drawn from each are this round's final subset, with 200 each for development/selection run. NQ paired inputs have been exported, while FEVER only generated query-only C1 tasks; at that freeze commit the matching index, BGE revision and new FEVER C1 were still to be completed, CPU regression 665 passed, and T2 was not claimed fully ready. The NQ Train ID/offset/corpus resource format also needs to be integrated in T3.

2026-10-24: [retrieved the NQ Train original corpus/index and restored the accompanying files](NQ_TRAIN_INDEX_RECOVERY.md); 64 independent BGE vector comparisons passed, latest CPU regression 681 passed. The difference between the verified snapshot and the historical C1 revision is kept in the restoration record; FEVER and Controller provider integration still proceed per the original T2/T3 to-dos, and the original frozen samples, held-out pools and algorithm prompts are unchanged.

Subsequently completed [full NQ location verification, Train provider integration and FEVER resource import](CONTROLLER_RESOURCE_ADAPTER_V1.md). NQ adds the `--nq-resources` entry point; the FEVER original package and full ID/original-text ordering have been accepted, while the FEVER dynamic provider, new C1 and the remaining T3 groups are still to be completed. The old interface notes and historical acceptance numbers below are kept; for the new verification of resource integration see the link.

Most recently completed: [FEVER provider and the new 900-item C1 collection package](CONTROLLER_FEVER_C1_V1.md); repository-wide 726 CPU regression items pass; the Top5 IDs/ordering of the fixed 10 historical queries are all reproduced. FEVER dynamic retrieval enhancement can now be enabled via `--fever-resources`, and the actual GPU status of the new C1 is registered separately. The FEVER unsupported description below corresponds to the initial merge version; the Full/No Probe/human-type groups and the formal seal are still pending T3.

## 1. Sources and merges

A new branch was created from E branch `5da1fe5`, merging in turn:

| Merge source | Source commit | Contents |
|---|---|---|
| `main` | `058f10b` | Already an E ancestor; Reference RAG, NQ Train pipeline |
| `reranking` | `338ff34` | Contains Diagnosis/Probe/Controller `7c7863f`, plus evidence reranking |
| `feature/retrieval-enhancement` | `1d97780` | Frozen NQ P3-20 retrieval enhancement |
| `feature/fever-c1` | `6ebdfc9` | Contains NQ C0/C1 `bcb248c`, C3 `8b03009`, and FEVER C1 |
| `feature/F500-c0c3` | `1238a4f` | FEVER C0/C3, index provenance |
| `feature/-ce-70B` | `6f08f2f` | 70B control, offline judge and results |
| `feature/verification-answer-repair` | `5da1fe5` | All of E's historical code, experiments, reviews and handovers |

All 10 remote branches are ancestors of the integration commit. Git blobs of `data/`, `results/`, `artifacts/`, `models/`, `configs/splits/` were compared branch by branch: committed data and experiment results at their original paths all keep their original bytes; see [branch_manifest.json](../artifacts/controller/integration_v1/branch_manifest.json) for details. The original branch histories were not rewritten.

Merge conflict handling:

- README merges the entry points, with old progress explicitly marked as historical record.
- `src/generation/qwen.py` keeps E's verified revision pinning, explicit NF4 GPU placement and output stopping logic; the 70B branch's old automatic CPU offload modification is not enabled. That branch's original code remains in its commit history, and the old offload environment-variable notes in `docs/LOCAL_ENV.md` do not apply to this integration version.
- The reranking branch's eager package-level imports would break the independence of the retrieval module; they were changed to lazy exports, keeping public class and function names.
- The FEVER branch moved pure prompt functions to `src/generation/prompts.py`; three retrieval tests now check against the actual definition site, and the prompt bytes are unchanged.
- The r4 frozen AST test is compatible with Python 3.12's empty `type_params` field; the original frozen fixture and prompt hashes are unchanged.

## 2. New execution path

```text
Frozen C1 JSONL / pure DiagnosisInput JSONL
  → Whitelist adaptation (no gold or human labels passed to runtime)
  → Frozen Diagnosis → optional single Probe → original Safe Binary
  → no_action / retrieval_enhancement / evidence_refinement / generation_repair
  → Unified results, call receipts, fallback, budget and cost
  → Separate process loads gold for offline evaluation
```

| File | Responsibility |
|---|---|
| `src/controller/execution.py` | Unified executor, strict input validation, per-module adaptation, shared budget, per-call traces |
| `src/controller/providers.py` | Loads and shares one Qwen externally; CPU reranker and optional NQ retrieval resources |
| `configs/controller/integration_v1.json` | Explicitly selects the E strategy and call/output budgets |
| `scripts/run_controller_integration.py` | Pre-check, CPU smoke, real runs, per-item saving and run manifest |
| `scripts/evaluate_controller_integration.py` | Joins gold independently after prediction, computes fixes, harmful repairs, net gain, cost |
| `tests/test_controller_integration.py` | Four-way execution, Probe overrides, failures and budget, gold isolation, shared model and CLI |
| `tests/test_controller_evaluation.py` | Paired samples, missing/duplicate references, fix and harmful-repair scoring |

The original `run_controller()` and each module's public interface are kept. This round does not change Diagnosis features, classifier parameters, the Probe prompt, Safe Binary thresholds, the P3-20 ranking formula, reranking's default keep5 or the E prompts.

### Explicit version selection for E

The integration configuration uses NQ `short_answer_quote` (the existing v7 Qwen short-answer-and-quote strategy) and FEVER `verify_repair` (the existing Qwen verify-and-repair strategy). Both use the same Qwen model object; only the output format and stop adaptation differ. This combination is an **explicit baseline for integration comparison**; it is not a reproduction of the v7 FEVER NLI results, nor a new performance-improvement conclusion.

The historical NLI experiments and v9 code are fully kept. Passing v9's synthetic contract cannot yet replace real quality acceptance, so it is not automatically enabled. The current source's `conservative_repair` is **r4**; changing the configuration to this name cannot be used to claim the frozen r3 is running. To reproduce r3, use the original frozen package.

### Execution and fallback

- `no_action` does not call an enhancement provider; if Probe was triggered earlier, the Probe cost is still counted.
- NQ P3-20 uses original-query Top20, revision-pinned BGE reranker raw logits, RRF, Top5, with at most one re-answer.
- FEVER's P3-20 request returns `unsupported_dataset`, does not call that route's provider, and keeps C1; it must be listed separately in coverage.
- Reranking keeps the original default keep5; if the order is unchanged there is no re-answer.
- When E has no valid output, the original module's fallback is used. A quote/short answer appearing in the evidence only proves the source matches and cannot be treated as semantic correctness.
- Provider exceptions count as attempted calls, saving the error and the C1 fallback; a Probe exception produces an uncertain result without overriding the original type. A classifier load/inference exception aborts the run and marks it failed, without forging a correct label.
- At most 4 Qwen calls (including Probe), 1 retrieval enhancement and 1 reranker API call per question. Actual counts are per provider call and do not equal the number of internal model batches.
- Record the full prompt seen by the model, the prompt hash, the generation result, evidence, diagnosis, final action and exceptions; these traces are not claimed to be the model's internal reasoning.
- Unknown token counts are kept as `null`; failed calls are not recorded as zero cost. Timing is explicitly the increment after the cached C1; full-system cost still needs the common C1 cost added, and load time is listed separately.
- If the C1 answer is kept, `final_evidence` keeps the C1 context; attempted enhancement contexts are saved separately in the module trace/call records.

## 3. Running

Use a separate environment. `requirements-controller.txt` reuses the merged team dependencies and pins the sklearn used by Diagnosis; install CUDA PyTorch according to the GPU platform. Do not mix with the pinned Transformers versions of historical experiment packages. CPU regression environment versions are in the verification archive; it does not mean real CUDA loading has been verified.

```bash
python -m pip install -r requirements-controller.txt
python scripts/run_controller_integration.py \
  --smoke --output-dir /tmp/controller-smoke-new
python -m unittest discover -s tests -v
```

Smoke has only 2 fictitious inputs, runs 5 modes, 10 results in total; the real frozen classifier and real routing/enhancement code are connected to fake providers. It is not a new NQ/FEVER benchmark, and is not used to tune semantic strategies.

The input may be the original standard C1 export (including `condition`, `model_name`, `retrieved_top5_ids`, etc.), or runtime JSONL containing only `sample_id,dataset,query,evidence,initial_answer`. Each item strictly requires an ordered Top5, finite retrieval scores, non-empty text and a unique ID. The NQ Train delivery should first be converted to this interface; the initial (AI) label workbook is never passed directly to the model.

Check on development data first, without opening the final test set:

```bash
python scripts/run_controller_integration.py \
  --input /path/to/frozen_dev_c1.jsonl \
  --preflight --output-dir /path/to/new_preflight
```

Formal comparison on CUDA (resource paths must belong to the same corpus/index version as this batch of C1):

```bash
python scripts/run_controller_integration.py \
  --input /path/to/frozen_dev_c1.jsonl \
  --nq-index /path/to/matching_nq.index \
  --nq-metadata /path/to/matching_nq_metadata.jsonl \
  --mode c1 --mode adaptive --mode always_retrieve \
  --mode always_rerank --mode always_repair \
  --output-dir /path/to/new_controller_run
```

NQ adaptive/always_retrieve must be given the index and metadata explicitly in real runs; the old dev index will not be quietly used for Train samples. Structure and selected ID/text are validated, and resource SHA-256s go into the environment record; **these hashes do not automatically prove that the index matches the original C1's version**, and the operator still needs to verify the delivery source. FEVER-only does not need the NQ index, but retrieval enhancement still does not support FEVER.

Each run requires a new output directory, flushing JSONL item by item. It produces `manifest.json`, `environment.json`, `predictions.jsonl`, `summary.json`; runs containing provider failures or budget overruns return a non-zero exit code. Resuming half-finished runs is not automatic for now, to avoid mixing results across code versions.

Offline evaluation: each gold JSONL item contains `sample_id,dataset,gold_answer_or_label`; for NQ it may be a list of answer strings, for FEVER SUPPORTS/REFUTES.

```bash
python scripts/evaluate_controller_integration.py \
  --predictions /path/to/new_controller_run/predictions.jsonl \
  --gold /path/to/dev_references.jsonl \
  --output /path/to/new_controller_run/offline_evaluation.json
```

Missing/duplicate gold, duplicate predictions, unpaired samples across modes or inconsistent C1 inputs for the same question all raise errors. NQ EM/F1 and FEVER label accuracy are kept separate; fix count, harmful repair count, net gain, action coverage and cost are reported together. NQ EM cannot replace semantic human review; this scoring also does not equal evidence-faithfulness acceptance.

## 4. Verification records

Full CPU regression: **635 passed**, see [tests.log](../artifacts/controller/integration_v1/tests.log); environment in [cpu_test_environment.json](../artifacts/controller/integration_v1/cpu_test_environment.json). The 8 compatibility failures from the first post-merge round are also kept in `initial_merge_tests.log`, and were not handled by skipping tests.

The branch provenance check covers 764 unique committed data/result/model/split file paths (excluding data READMEs and placeholder files), all consistent with the corresponding Git blobs of their source branches. The CPU smoke run manifest and per-item outputs are archived separately; these are only for engineering verification and do not report QA improvement.

## 5. Current scope of completion and next steps

Completed: full-branch merge, C1 post-processing executor, two-dataset interfaces, traceable fallback, shared call budget, paired offline scoring, CPU tests and fictitious smoke. Going from raw questions to C1 continues to use the existing Reference RAG runner, to avoid recomputing or changing the frozen C1 in this integration.

Still to be completed, in priority order:

1. T2 freezes, per the T1 protocol, the same batch of development/selection-run/final held-out data, paired gold, sample counts, exposure registration and matching resources; there are as yet no post-integration QA gain numbers.
2. T3 adds offline human-type triggering, No Probe and failure type × operation analysis, separating routing loss from module loss. Fixed-action results are reused, and human labels do not enter online model prompts.
3. T3 completes FEVER retrieval enhancement; if resource limits allow only a partial scope, it must be declared before seeing results, and fallback cannot be counted as implemented.
4. T3 implements Full in the frozen order and budget; after completing scoring/complete-list acceptance and the execution seal, it enters the T4 GPU comparison.
5. Based on the same-batch comparison, optimise only the largest bottleneck, with at most one round of pre-registered candidates. Keep the final unseen test set; distillation, DPO, knowledge-base modification and RSI are not implicit changes of this integration.

Pre-merge repository-wide and data-quality conclusions are in the [23 October audit](PROJECT_AUDIT_AND_RSI_PLAN_20261023.md). The "not yet integrated" description in the audit corresponds to the pre-merge state; this file records the engineering progress after it.
