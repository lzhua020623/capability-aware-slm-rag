# Diagnosis + Probe Interface

Frozen runtime interface for the Diagnosis layer. Code: `src/diagnosis/labels.py`, `src/diagnosis/schema.py`.

The classifier, Probe, and Controller are not implemented yet. This document fixes only the labels and the input/output contract.

## Runtime model

- `Qwen/Qwen2.5-7B-Instruct` (4-bit NF4, greedy) is the host RAG generator. It is referred to conceptually as **Qwen-A**.
- Diagnosis itself must not depend on a larger model.
- Probe, when implemented later, may make at most one additional call to the same Qwen-A in the initial design.
- Larger-model controls from the preliminary experiment (32B / 70B / 72B) are offline-only. They are not part of the runtime architecture.

## Diagnosis runtime input

`DiagnosisInput`:

| Field | Type | Meaning |
|---|---|---|
| `sample_id` | `str` | Sample ID. Integer IDs (e.g. FEVER) are converted to strings. |
| `dataset` | `DatasetName` | Canonical dataset name: `nq` or `fever` only. Source aliases such as `natural_questions` must be mapped by the adapter before building the input. |
| `query` | `str` | NQ question or FEVER claim. |
| `evidence` | `list[EvidenceItem]` | Retrieved evidence shown to Qwen-A, in rank order. |
| `initial_answer` | `str` | Qwen-A's initial answer for the query and evidence. |

`EvidenceItem`: `rank` (`int`, 1-based, unique per input), `passage_id` (`str`), `text` (`str`), `score` (`float`, retrieval score).

## Runtime-forbidden information

The following cannot be used as Diagnosis runtime inputs or features:

- gold answers and gold labels
- gold evidence, including gold passage/page IDs and whether gold evidence is in the Top-5
- correctness, EM, F1, and accuracy references
- C0 and C3 outcomes
- larger-model outcomes (32B / 70B / 72B answers or judge verdicts)

These may be used only for offline labelling and evaluation. `src.diagnosis.schema.FORBIDDEN_RUNTIME_FIELDS` lists the field names that the runtime schema must never contain.

## Diagnosis output

`DiagnosisOutput`:

| Field | Type | Meaning |
|---|---|---|
| `diagnosis_label` | `DiagnosisLabel` | One of the four labels below. |
| `confidence` | `float` | Confidence in `diagnosis_label`, in `[0.0, 1.0]`. |
| `needs_probe` | `bool` | Whether Diagnosis requests a Probe call. The Probe threshold is not fixed yet. |
| `class_scores` | `dict[DiagnosisLabel, float] \| None` | Optional per-label scores for debugging and evaluation. The Controller must not depend on it. |

## Four labels

| Label | Definition |
|---|---|
| `correct` | The initial answer is acceptable; no failure. |
| `evidence_insufficiency` | The retrieved evidence lacks the information needed to answer the query. |
| `evidence_accessibility` | The needed information is in the retrieved evidence, but noise, conflict, or ordering prevents it from being used. |
| `evidence_utilisation` | The retrieved evidence is clear and sufficient, but Qwen-A still answers incorrectly. |

"7B oracle recovered", "likely small-model capability limitation", and "unresolved hard case" are offline analysis categories. They are not runtime Diagnosis labels.
