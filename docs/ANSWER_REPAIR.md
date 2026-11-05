# Verification and Answer Repair for E

E corresponds to Evidence Utilisation: the model has already seen usable evidence, yet the initial answer may pick the wrong entity, misread a relation, ignore a negation, or output an answer not supported by the evidence. Diagnosis is owned by other members, and the Controller decides when to call E.

As of 2026-10-16, the working basis is v7; the next round, v8, is not yet implemented. The v7 main bundle and batch 4 have completed one GPU run, and the 541+300 results passed replay with their respective frozen code; neither strategy was promoted to default, and human semantic acceptance is still pending. The current entry points are the [handoff checklist](ANSWER_REPAIR_HANDOFF.md), [v7 results](ANSWER_REPAIR_RESULTS_20261014_V7.md) and the [combined review sheet](../artifacts/answer_repair/v7_combined_review/review_all.txt).

Of the 37 confirmed early candidates, 19 core error candidates and 9 correct controls have been used for development, and 9 are deferred. User confirmation of the key v6r1 cases is in the [confirmation checklist](ANSWER_REPAIR_CONFIRMATION_20261014.md). These existing development bases can be used for engineering improvements; new semantic rules still require review of the key disputed cases and separate acceptance.

The text below explains the shared interface and the early single-pass JSON verification behaviour; for the specific behaviour of `short_answer_quote` and `fever_nli_anchored`, the [v7 protocol](ANSWER_REPAIR_V7.md) is authoritative, and the independent model cost is listed separately. Old designs, failure analyses and execution records are in the [history archive](history/answer_repair/README.md); the old "pending run" status must not be taken as current progress.

## Inputs, outputs and processing rules

`repair_answer(DiagnosisInput, generator)` directly reuses the [shared interface](DIAGNOSIS_PROBE_INTERFACE.md). The input contains only the sample ID, dataset, query, original evidence and initial answer. The adapter selects fields explicitly and does not pass gold, C0/C3, correctness, retrieval gold hit or large-model results to the model.

Evidence content and order are kept exactly as in C1. The model outputs:

```json
{"verdict":"repair","answer":"Paris","citations":[{"rank":1,"quote":"The capital of France is Paris."}]}
```

| Verdict | First-version behaviour |
|---|---|
| `keep` | Cite evidence supporting the original answer; the original answer must remain unchanged |
| `repair` | Provide a repaired answer and citations; for NQ the new answer must be a complete text span from the citation, for FEVER it must be exactly SUPPORTS or REFUTES |
| `insufficient` | Do not force an answer change; record the evidence-insufficient status and let the upper layer handle unresolved or follow-up actions |

The cited rank must exist, and the quote must appear in the corresponding passage; matching tolerates case and whitespace differences and preserves words and punctuation. Format errors, fabricated citations, illegal labels or unsupported new NQ spans all fall back to the initial answer and record `invalid_output`. Backend exceptions stop the run directly and can be resumed; they are not disguised as the model choosing keep. With no usable evidence there are zero calls.

This example only explains the interface and is not an experimental result. Real NQ C1 also contains similar task-understanding errors; for example, the question asks "what is the sentencing reform act of 1984", but the initial answer gives "U.S. Sentencing Commission", while the existing evidence describes both the law and the commission it established. E should check the object the question asks about and the relations in the evidence before choosing an answer span, not merely check that both entities appear.

A citation check proves that the span comes from the input; it cannot prove that the span entails the answer. FEVER in particular requires checking entities, relations, negation, quantities and multi-evidence combinations. The first-version log fixes the marker `independent_support_status=not_checked` and does not treat a valid citation as proof of NLI entailment, faithfulness or runtime answer correctness. The explicit independent MNLI branch of v6r1 has been tested, but dev50 showed two harmful repairs caused by wrong entities; high scores are still not human ground truth, and supportiveness and calibration still need later acceptance.

## Boundaries with Diagnosis and the Controller

Final integration uses `generation_repair(request, decision, generator)` in `src/compensation/controller_adapter.py`. `decision` is the `ControllerDecision` produced after upstream Diagnosis, Probe and Safe Binary have completed; E only executes action=`generation_repair`, and other actions return the original answer with zero calls. Even if the upstream confidence is below the old threshold, E does not re-judge it; routing policy is the Controller's responsibility. The caller should reuse the same loaded generator.

The offline runner uses `--controller-decisions path.jsonl` to receive the seven fields of `ControllerDecision.to_dict()`; the file should cover all IDs in this evaluation. Unknown fields, missing decisions or mismatched IDs all raise errors; an optional dataset field is supported. Example:

```json
{"sample_id":"1","action":"generation_repair","diagnosis_label":"evidence_utilisation","diagnosis_confidence":0.2,"used_probe":true,"probe_overrode_diagnosis":false,"needs_compensation":true}
```

After integrating the upstream Controller/Probe branch, the model can be loaded only once, injecting the same model's raw greedy call into the Probe and giving E the generator with a JSON stopper. The Probe's prompt, 160-token budget, trigger and Safe Binary are unchanged:

```python
from src.compensation.qwen_repair import QwenRepairGenerator
from src.compensation.controller_adapter import generation_repair
from src.controller.controller import run_controller
from src.probe.probe import EvidenceProbe, PROBE_MAX_NEW_TOKENS
from src.generation.qwen import generate_greedy

e_generator = QwenRepairGenerator()
probe = EvidenceProbe(lambda prompt: generate_greedy(
    e_generator.tokenizer, e_generator.model, prompt,
    max_new_tokens=PROBE_MAX_NEW_TOKENS,
)[0])
decision = run_controller(item, classifier, probe)
e_result = generation_repair(item, decision, e_generator)
```

This is integration usage; the current E branch does not copy the upstream Controller/Probe implementation. The four actions and the real upstream `ControllerDecision` class have completed deterministic fixture verification in an isolated directory; a real GPU Controller→E development trial run has also been completed, with 27 Probe calls and 1 E call and no net gain, see the [standalone experiment report](history/answer_repair/ANSWER_REPAIR_RESULTS_20261013.md). The external model loader, classifier and input objects are all provided by the caller. The other two compensation types were not executed in that trial run, so it cannot be called a complete adaptive system evaluation.

The old `--diagnoses` is used only for standalone development ablations and cannot replace final Controller routing. It accepts runtime output JSONL as below, with dataset using `nq`/`fever`, and covers all IDs in this evaluation:

```json
{"sample_id":"1","dataset":"nq","diagnosis_label":"evidence_utilisation","confidence":0.85,"needs_probe":false}
```

E is executed only when the primary label is utilisation and `needs_probe=false`; other labels keep the initial answer with zero calls, and samples waiting for the Probe are also skipped. Thresholds and the Probe policy are managed by the Controller; E does not set its own confidence threshold. A missing diagnosis raises an error.

`--all` is an explicit Always-Verify development ablation; it does not require a diagnosis classifier to exist and does not read gold to pick error samples. It can be used to verify E's effect first, before connecting the team's real diagnosis output. When human labels are used for offline oracle-routing ablations, they must be named and reported separately and cannot be treated as runtime diagnosis.

## Files

| File | Purpose |
|---|---|
| `src/compensation/answer_repair.py` | Pure-Python strategy entry point, verification rules and fallback |
| `src/compensation/qwen_repair.py` | Lazily loads the shared Qwen NF4 generator, records tokens and latency |
| `src/compensation/controller_adapter.py` | Receives the final ControllerDecision and executes only the E action |
| `scripts/run_answer_repair.py` | Reads C1, optional routing, pre-check, persistence, locking and checkpoint resume |
| `src/evaluation/answer_repair.py` | Standalone offline paired scoring and aggregation |
| `tests/test_answer_repair.py` | Tests for behaviour, information isolation, routing and interruption recovery |

The code requires Python 3.11+, consistent with diagnosis's `StrEnum` interface. Pre-checks and tests need only the standard library and do not load torch, transformers, FAISS, models or datasets.

## Directly runnable on the current computer

From the repository root:

```bash
python3.11 scripts/run_answer_repair.py --dataset nq --source-ref origin/feature/-ce-70B --all --check-only
python3.11 scripts/run_answer_repair.py --dataset fever --source-ref origin/feature/-ce-70B --all --check-only
python3.11 -m unittest discover -s tests -v
```

`--source-ref` reads archived C1 directly from that commit, including the recorded Top-5 full text. No re-retrieval or download of Wikipedia/index is needed. Alternatively, `--c1 path/to/c1.jsonl` reads a local C1 handoff file. `--dry-run` prints the first prompt, still without calling the model or saving predictions, to avoid mixing simulated output into real results.

## Running on a GPU machine

### First-round portable experiment bundle (2026-10-13)

This machine is an Apple Silicon Mac without PyTorch or Transformers installed and without CUDA; real repair runs on Colab. The preparation script freezes the development samples and packages the current code and the existing C1 full text. The command below builds the first version; an existing version bundle refuses to be overwritten by different code:

```bash
python3.11 scripts/prepare_answer_repair_experiment.py
```

Revised versions reuse the original development IDs and build a separate bundle:

```bash
python3.11 scripts/prepare_answer_repair_experiment.py --out-dir artifacts/answer_repair/v2 --bundle-name answer_repair_v2_gpu
```

This generates `configs/splits/answer_repair_v1_dev.json` and `artifacts/answer_repair/v1/answer_repair_v1_gpu.tar.gz`. After extraction, follow `README_GPU.md` inside the bundle; no clone, raw-data download, retrieval-index rebuild or manual stitching of uncommitted code is needed; the model still needs to be downloaded to the Hugging Face cache on first run.

Samples are taken only from the intersection of training IDs in Diagnosis v1 and v2-expanded, excluding all validation/test IDs of both versions. The 90 NQ train IDs of v1 became validation/test in v2, so the whole v1 train can no longer be used as E's development pool. The intersection offers 331 FEVER and 214 NQ samples. Using SHA-256 ordering with fixed seed 42, 50 are selected from each, without using correctness, reference answers or diagnosis labels to select samples. Reference answers are kept only for the existing offline scoring.

In the first-round FEVER development set, 45 initial answers are correct and 5 wrong; in NQ, 20 are correct by initial EM and 30 wrong. The two smoke sets have 10 each, being the first 10 of each development set, and cannot be added to the development results as independent samples. First run the FEVER smoke and review the output, then run the Always-Verify and equal-output-budget Reanswer comparisons on both development sets. This small-scale result is only for development and for judging whether to expand, and cannot serve as the final held-out result.

The GPU environment needs Python 3.11+ and a CUDA PyTorch compatible with the host. With CUDA PyTorch already present, the remaining E dependencies need only `requirements-answer-repair.txt`, without FAISS or sentence-transformers:

```bash
python -m pip install -r requirements-answer-repair.txt
python scripts/check_answer_repair_environment.py --require-cuda --out environment.json
```

The environment check does not load or download the model. Real call logs additionally record Python, dependency versions, CUDA, GPU, the revision obtainable from the model config, the tokenizer revision and the device map; if the revision is empty, the actual model snapshot must be verified separately. When resuming from a checkpoint, a changed environment record refuses further writes to the same file. The portable bundle includes a SHA-256 checksum list of all input and code files.

### Running from the repository

Alternatively, clone the `feature/verification-answer-repair` branch on the GPU machine and install CUDA PyTorch, the project dependencies and the correct Python version per [SETUP.md](SETUP.md). Use the same frozen 7B, NF4 and greedy; do not connect the offline 70B backend. The frozen input bundle, results and CPU replay steps for a fresh clone are in the [handoff checklist](ANSWER_REPAIR_HANDOFF.md).

The following shows the general invocation for archived C1. First-round development actually used the training-intersection samples in the portable bundle; without verifying the split, do not use the whole 500 as the development set for prompt tuning. Engineering smoke results use a separate file:

```bash
python scripts/run_answer_repair.py --dataset fever --source-ref origin/feature/-ce-70B --all --limit 10 --out results/answer_repair/fever_smoke10.jsonl
```

Check JSON parsing, citation validation, context length and output tokens. Formal run:

```bash
python scripts/run_answer_repair.py --dataset fever --source-ref origin/feature/-ce-70B --all
python scripts/run_answer_repair.py --dataset nq --source-ref origin/feature/-ce-70B --all
```

After connecting diagnosis, use a separate output file:

```bash
python scripts/run_answer_repair.py --dataset fever --source-ref origin/feature/-ce-70B --controller-decisions path/to/controller_decisions.jsonl --out results/answer_repair/fever_routed.jsonl
```

Each `--limit` denotes a fixed development cohort of the first N items; changing N requires a separate output file. Formal comparisons must use the same ID split and cannot pick samples only from failures that are easier to repair.

The default repair output budget is 384 tokens, because JSON and short citations need extra output space; the original C1 64-token setting and results are not modified. There is also `--strategy reanswer`, which regenerates directly with the same evidence and C1 instruction, as an extra call/output-budget control; it uses the same 384-token cap as E. Under greedy decoding, output for the same prompt often stays identical; differences need to be examined together with truncation, model revision and environment checks, and cannot be automatically attributed to the repair algorithm.

The software environment of the shared model loader, the remote model revision and GPU characteristics still affect reproduction. At run time, save the Git commit, Python package versions, model revision, GPU and time. The script fingerprint prevents mixing writes from different inputs, code and settings, but cannot replace these environment records.

## Interpreting results and cost

E results are written to `results/answer_repair/` with `condition=E`; `C2` in the proposal refers specifically to accessibility-controlled evidence, and this name is not reused. Each record saves the initial/final answer, evidence IDs, actual citations, raw output, parse status, extra calls and tokens/latency, and source points to the frozen C1 input.

Citation anomalies or insufficient evidence both keep the original answer, so the status is not "repair succeeded". Offline scoring reads reference answers separately, using the existing NQ EM/F1 and FEVER scorers; it does not participate in the model prompt or action routing.

`.summary.json` aggregates all evaluated samples, including samples skipped by routing:

- `fixed`: initially wrong → finally correct; `regressed`: initially correct → finally wrong.
- `net_correct_gain = fixed - regressed`; `accuracy_gain` is its proportion of the total sample count.
- The denominator of `repair_success_rate` is the number initially wrong; the denominator of `regression_rate` is the number initially correct.
- Counts for all and changed samples, status distribution, NQ F1, paired McNemar exact p-value.
- Extra calls, total tokens and extra generation latency; `output_tokens_estimate` comes from re-tokenizing the output text and excludes EOS.

Extra generation latency excludes model loading and overall framework time. Without a GPU unit price, no monetary cost is fabricated. Archived C1 may come from a different environment, so its latency cannot be used directly to derive an end-to-end speed-up ratio for the same deployment; a complete system evaluation needs timing in the same environment and must also count the cost of diagnosis/NLI/Probe.

## Resuming

Each result is appended, flushed and fsynced. An output lock ensures only one writer process per file. After an interruption, continue with the same command; it checks the fingerprint, ID order, initial answers and evaluation consistency, then skips completed records.

A missing newline at the end of a complete final record is added automatically; incomplete JSON, duplicate/out-of-order results, a different source or different prompt/token settings refuse to resume and keep the original file. A hard interruption may leave a `.lock` behind; remove it only after confirming the original process has stopped. Use a separate file when experimental conditions or inputs change.

## Experiment completion criteria

Per the latest instruction of 2026-10-14, human re-evaluation of results may be deferred; items awaiting review are isolated, and automated engineering work continues. Engineering completion does not mean that algorithm effectiveness or team-wide held-out acceptance is complete.

The first round first reports C1 vs Always-Verify paired results on a fixed development set containing both correct and wrong answers, reviews all changed and invalid outputs, and then decides whether to change the prompt. Human review should cover cases repaired correctly, broken, self-declared keep by the model, insufficient evidence, and valid quotes that do not support the conclusion.

The NQ main table continues to keep frozen EM/F1; semantic re-evaluation must use the same judge/human rules for both initial and final answers, and must not score only repaired answers leniently. Independent verifier/human faithfulness results are reported separately.

Afterwards, compare Always-Verify with diagnosis-routed E on held-out samples, and form a failure-type × action matrix together with the other two compensation operations. Whether to keep E in the end, and how to trigger E, should be decided by net gain, harmful repair rate and cost results; the current engineering tests do not provide this effectiveness conclusion.
