# Diagnosis + Evidence Probe + Controller: Integration Handoff

Audience: the three Enhancement owners (`retrieval_enhancement`, `evidence_refinement`, `generation_repair`).

Status: **frozen**. Diagnosis v2-expanded, the Evidence Probe prompt, the Probe trigger, the Safe Binary rule and the Controller mapping were selected on the validation split and evaluated once on the test split. They must not be retrained, re-tuned or changed for integration.

Label definitions and the `DiagnosisInput` contract are in `docs/DIAGNOSIS_PROBE_INTERFACE.md`.

## A. Final frozen pipeline

```text
Reference RAG output (Qwen2.5-7B-Instruct NF4, BGE Top-5, C1)
  -> Diagnosis v2-expanded (hierarchical logistic regression)
  -> selective Evidence Probe (one extra Qwen2.5-7B call, only when triggered)
  -> Safe Binary resolution
  -> Controller routing (ControllerDecision.action)
  -> Enhancement module (owned by teammates; not implemented here)
```

| Component | Frozen setting | Code / artifact |
|---|---|---|
| Diagnosis | Stage 1: C=10, balanced, threshold 0.50; Stage 2: C=0.01, balanced | `src/diagnosis/hierarchical_classifier.py`, `models/diagnosis/diagnosis_v2_expanded_hierarchical.json` |
| Probe trigger | `V2_EXPANDED_TRIGGER = TriggerPolicy(low=0.50, high=0.65, stage2_threshold=0.50)` | `src/probe/trigger.py` |
| Evidence Probe | Qwen/Qwen2.5-7B-Instruct, 4-bit NF4, `do_sample=False`, at most one call per sample | `src/probe/probe.py`, `src/probe/prompts.py`, `src/probe/schema.py` |
| Safe Binary | A Diagnosis failure becomes `correct` only if the Probe says `correct`, Stage 1 P(failure) <= 0.70 and max Stage 2 score <= 0.50. Otherwise the Diagnosis label (including its failure type) is kept. | `src/controller/controller.py` (`SAFE_OVERRIDE_MAX_STAGE1`, `SAFE_OVERRIDE_MAX_STAGE2`, `resolve_final_diagnosis`) |
| Controller | Label -> action mapping below; routing only | `src/controller/controller.py` (`ACTION_BY_LABEL`, `route_controller`, `run_controller`) |

The Probe's typed labels (insufficiency / accessibility / utilisation) never change the Diagnosis failure type; only the Safe Binary override to `correct` is applied.

## B. Frozen interfaces

Enhancement code must consume `ControllerDecision` (and the same `DiagnosisInput`). It must **not** re-implement, re-train or bypass Diagnosis, the Probe, the trigger or Safe Binary.

`ControllerAction` (`src/controller/schema.py`, `StrEnum`):

| Value | Meaning |
|---|---|
| `no_action` | Keep the original Reference RAG answer. |
| `retrieval_enhancement` | Routed to the retrieval enhancement owner. |
| `evidence_refinement` | Routed to the evidence refinement owner. |
| `generation_repair` | Routed to the generation repair owner. |

Mapping (`ACTION_BY_LABEL`):

| Final diagnosis label | Action |
|---|---|
| `correct` | `no_action` |
| `evidence_insufficiency` | `retrieval_enhancement` |
| `evidence_accessibility` | `evidence_refinement` |
| `evidence_utilisation` | `generation_repair` |

Entry points:

```python
from src.controller.controller import run_controller            # Diagnosis -> optional Probe -> Safe Binary -> routing
from src.controller.schema import ControllerAction, ControllerDecision
from src.diagnosis.adapter import c1_record_to_input             # C1 Reference RAG record -> DiagnosisInput
from src.diagnosis.hierarchical_classifier import HierarchicalDiagnosisClassifier
from src.probe.probe import EvidenceProbe

classifier = HierarchicalDiagnosisClassifier.load("models/diagnosis/diagnosis_v2_expanded_hierarchical.json")
probe = EvidenceProbe.from_qwen()                                # loads Qwen2.5-7B NF4 once
decision = run_controller(item, classifier, probe)               # item: DiagnosisInput; default trigger = V2_EXPANDED_TRIGGER
```

`run_controller` returns only a `ControllerDecision`. If an owner also needs the raw Probe output, use the same frozen path in two steps: `probed = diagnose_with_probe(item, classifier, probe, V2_EXPANDED_TRIGGER)` (`src/probe/pipeline.py`) and then `decision = decide_from_probed(probed, sample_id=item.sample_id)` (`src/controller/controller.py`). This produces the same decision.

## C. Runtime information available to Enhancement

From the `DiagnosisInput` passed to the Controller (`src/diagnosis/schema.py`):

| Field | Type | Meaning |
|---|---|---|
| `sample_id` | `str` | Sample ID (logging / joining only; must not change behaviour). |
| `dataset` | `DatasetName` | `nq` or `fever`. |
| `query` | `str` | NQ question or FEVER claim. |
| `evidence` | `list[EvidenceItem]` | Reference RAG Top-5 evidence in rank order. |
| `initial_answer` | `str` | Reference RAG answer (NQ short answer; FEVER label such as `SUPPORTS`). |

`EvidenceItem`: `rank` (`int`, 1-based), `passage_id` (`str`), `text` (`str`), `score` (`float`, retrieval score).

From `ControllerDecision` (`src/controller/schema.py`, frozen dataclass):

| Field | Type | Meaning |
|---|---|---|
| `action` | `ControllerAction` | Routed action. |
| `diagnosis_label` | `DiagnosisLabel` | Final label after Safe Binary resolution. |
| `diagnosis_confidence` | `float` | Confidence of the original Diagnosis decision, in [0, 1]. |
| `used_probe` | `bool` | Whether the Probe was triggered for this sample. |
| `probe_overrode_diagnosis` | `bool` | Whether Safe Binary changed a Diagnosis failure to `correct`. |
| `sample_id` | `str \| None` | Logging only; excluded from equality. |
| `needs_compensation` | `bool` | Derived: `action != no_action`. |

`ControllerDecision.to_dict()` serialises these fields.

Optional, only when the Probe was triggered (`ProbedDiagnosis.probe_result`, `src/probe/schema.py`): `ProbeResult` with `answer_supported`, `evidence_sufficiency`, `evidence_accessibility`, `supporting_ranks`, `conflicting_ranks`, `probe_label`, `evidence_verdict` (FEVER), `reason`, `raw_output`, `parse_error`. It is `None` for about two thirds of samples. Treat it as a hint, never as ground truth.

## D. Forbidden information

Enhancement runtime must NOT use:

- `target_label`
- gold answer
- gold evidence (gold passage / page IDs, gold evidence text, whether gold evidence is in the Top-k)
- `offline_evaluation` fields in any result file
- Test annotations (labels, `metadata`, `needs_manual_review`, review reasons)
- correctness, EM / F1, C0 / C3 outcomes, oracle or larger-model (32B / 70B / 72B) results

These are offline-only and may be used only for evaluation after all predictions are produced. `src.diagnosis.schema.FORBIDDEN_RUNTIME_FIELDS` lists field names that runtime schemas must never contain. Do not select thresholds, prompts or models on the test split.

## E. Responsibilities

| Owner | Responsibility |
|---|---|
| Diagnosis + Probe owner (this handoff) | Diagnosis, Evidence Probe, Safe Binary resolution, Controller routing |
| Teammate 1 | `retrieval_enhancement` (for `evidence_insufficiency`) |
| Teammate 2 | `evidence_refinement` (for `evidence_accessibility`) |
| Teammate 3 | `generation_repair` (for `evidence_utilisation`) |

The Controller currently performs routing only. It imports no retriever, reranker, generator-repair or compensation code and executes none of the Enhancement actions.

## F. Known limitations (final PRIMARY unseen test, n = 473)

PRIMARY excludes one Probe prompt-development sample (NQ `3724356943291190911`) from the 474-sample test split. Source: `results/final/diagnosis_probe_controller_test_evaluation.json`.

| Metric | Diagnosis only | Diagnosis + Probe + Safe Binary |
|---|---|---|
| Accuracy | 0.615 | 0.691 |
| Macro-F1 | 0.348 | 0.370 |
| Failure Detection Recall | 0.701 | 0.619 |
| correct -> failure false positives (FP) | 121 | 81 |
| failure -> correct false negatives (FN) | 29 | 37 |

- Safe Binary override precision: 0.833 (40 of 48 overrides were truly `correct`).
- Probe coverage: 162 / 473 = 34.2% (one Qwen call per triggered sample).
- Safe Binary trades recall for precision: Failure Detection Recall drops from 0.701 to 0.619, so about 38% of true failures are routed to `no_action` and will not reach any Enhancement module.
- Diagnosis is weak at separating `evidence_accessibility` and `evidence_utilisation` (per-class F1 after Safe Binary: insufficiency 0.379, accessibility 0.051, utilisation 0.217).

Offline composition of each routed action on the PRIMARY test (true label of the routed samples, after Safe Binary):

| Action | Routed | true correct | true insufficiency | true accessibility | true utilisation |
|---|---|---|---|---|---|
| `no_action` | 332 | 295 | 16 | 6 | 15 |
| `retrieval_enhancement` | 64 | 34 | 22 | 0 | 8 |
| `evidence_refinement` | 29 | 15 | 10 | 1 | 3 |
| `generation_repair` | 48 | 32 | 4 | 3 | 9 |

Consequences for Enhancement owners:

- **Do not assume the Controller action is correct.** Many routed samples are actually `correct` (34 / 64, 15 / 29 and 32 / 48), and only 1 of 29 `evidence_refinement` samples is a true accessibility failure.
- Enhancement modules must tolerate misrouting. They must not degrade answers that were already correct, and they should be able to keep the original answer when they find no improvement.
- FEVER is routed only to `no_action` (42) or `retrieval_enhancement` (29). FEVER has no trusted accessibility samples.
- Report Enhancement gains together with harm on routed samples that were already correct.

## G. Integration contract

Interface only; the Enhancement functions are owned by teammates and are not implemented here.

```python
decision = run_controller(item, classifier, probe)

if decision.action is ControllerAction.NO_ACTION:
    final_answer = item.initial_answer                       # keep original result
elif decision.action is ControllerAction.RETRIEVAL_ENHANCEMENT:
    final_answer = retrieval_enhancement(item, decision)     # teammate 1
elif decision.action is ControllerAction.EVIDENCE_REFINEMENT:
    final_answer = evidence_refinement(item, decision)       # teammate 2
elif decision.action is ControllerAction.GENERATION_REPAIR:
    final_answer = generation_repair(item, decision)         # teammate 3
```

Each Enhancement function receives only the runtime `DiagnosisInput` and the `ControllerDecision` (and, if needed, the optional `ProbeResult`). It must not read any field listed in section D.

## Reproducibility files

| File | Content |
|---|---|
| `configs/splits/diagnosis_v2_expanded_split.json` | Frozen train / validation / test split |
| `results/diagnosis/diagnosis_v2_expanded_validation.json` | Diagnosis validation selection and metrics |
| `results/probe/probe_v2_expanded_validation.json` | Probe validation outputs |
| `results/probe/probe_v2_safe_override_comparison.json` | Safe Binary selection on validation |
| `results/controller/controller_v1_validation.json` | Controller validation dry-run |
| `results/diagnosis/diagnosis_v2_expanded_test.json` | Final test Diagnosis predictions |
| `results/probe/probe_v2_expanded_test.json` | Final test Probe outputs |
| `results/controller/controller_v1_test.json` | Final test Controller decisions |
| `results/final/diagnosis_probe_controller_test_evaluation.json` | Final test evaluation (offline) |
| `scripts/run_final_test_evaluation.py` | Final test preflight / inference / evaluation script |
