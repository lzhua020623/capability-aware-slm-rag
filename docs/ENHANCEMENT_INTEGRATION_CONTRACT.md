# Enhancement Integration Contract (P63/P64)

2026-10-27: the user selected this upstream design as the future baseline. The real-module bridge and strict production loader are described in [the current handoff](CONTROLLER_1024_BASELINE.md). Original runtime features and P60 model usage provenance remain required before inference. The following 10/26 note describes historical T4/T5 execution.

2026-10-26 integration note: this is an optional host contract. The current
frozen experiment still uses `IntegratedController` with its metered providers
and original Probe policy. P60 is not activated by this merge. See
[compatibility integration and activation conditions](DIAGNOSIS_138D52A_INTEGRATION.md).

Audience: owners of Retrieval / Evidence Refinement / Generation Repair modules.

Status: **interface + host pipeline frozen**. Diagnosis, frozen selection, and
Controller routing are upstream. Teammate Enhancements plug in **without**
changing Diagnosis or Controller.

## End-to-end host pipeline (P64)

```text
query
  -> Reference RAG  (injected; returns DiagnosisInput)
  -> HierarchicalDiagnosis
  -> FrozenOnlineSelectionAdapter.select(...)
  -> SelectionDecision
  -> Controller.decide(...) -> ControllerDecision
  -> EnhancementRequest
  -> EnhancementExecutor.execute(...)
  -> EnhancementResult
  -> AdaptiveRAGResult
```

Public entry: `src.pipeline.run_adaptive_rag` / `AdaptiveRAGPipeline`.

Controller **decides** the action. Executor **dispatches**. Enhancement **runs**.
The host does not hard-code teammate class names.

Probe is **OFF** on this path (`used_probe=False`).

## Canonical actions (repo names)

| ControllerAction | Conceptual alias | Typical failure_type |
|---|---|---|
| `no_action` | — | any / not selected |
| `retrieval_enhancement` | — | `evidence_insufficiency` |
| `evidence_refinement` | `accessibility_enhancement` | `evidence_accessibility` |
| `generation_repair` | `utilisation_enhancement` | `evidence_utilisation` |

## EnhancementRequest

Runtime-safe fields only (`src/compensation/schema.py`):

- `query`, `initial_answer`, `evidence` (`EvidenceItem` Top-k)
- `controller_decision` (`ControllerDecision`)
- `failure_type` (`DiagnosisLabel`)
- optional `sample_id`, `dataset`, `context` (no gold / offline keys)

```python
from src.compensation.schema import EnhancementRequest

request = EnhancementRequest.from_runtime(
    query=item.query,
    initial_answer=item.initial_answer,
    evidence=item.evidence,
    controller_decision=decision,
    sample_id=item.sample_id,
    dataset=item.dataset.value,
)
```

## EnhancementResult

- `action`, `final_answer`, `evidence`
- `changed_answer`, `status` (`success` | `no_action` | `unavailable` | `error`)
- `metadata`, optional `latency_ms`, optional `error`

`status=success` means the module executed without crashing — **not** that the
answer is correct.

Host pipeline maps Enhancement status to `PipelineStatus`
(`no_action` / `success` / `enhancement_unavailable` / `enhancement_error`).

## Interface

```python
from src.compensation.interface import Enhancement
from src.compensation.schema import EnhancementRequest, EnhancementResult

class MyModule:  # or adapter wrapping a teammate class
    def enhance(self, request: EnhancementRequest) -> EnhancementResult:
        ...
```

Logical roles (same protocol): `RetrievalEnhancement`,
`EvidenceRefinementEnhancement`, `GenerationRepairEnhancement`.

## Future teammate integration (minimal steps)

When a real module is ready:

1. Import the team module (do **not** change Diagnosis/Controller).
2. Write a thin adapter implementing `Enhancement.enhance`.
3. Map team I/O ↔ `EnhancementRequest` / `EnhancementResult`.
4. At host startup: `executor.register(action, adapter)`.
5. Run contract / host tests.
6. Leave Diagnosis, selection thresholds, and Controller routing untouched.

Adapters **may** forward injected retriever / generator / config and normalize
exceptions. Adapters **must not** rewrite teammate algorithms.

### Retrieval (`retrieval_enhancement`)

```python
class RetrievalEnhancementAdapter:
    def __init__(self, team_module, generate_fn=None):
        self.team = team_module
        self.generate_fn = generate_fn

    def enhance(self, request: EnhancementRequest) -> EnhancementResult:
        # Map request -> team input; call team; map output -> EnhancementResult.
        out = self.team.run(request.query, request.evidence, request.initial_answer)
        return EnhancementResult(
            action=request.controller_decision.action,
            final_answer=out.final_answer,
            evidence=tuple(out.evidence),
            changed_answer=out.final_answer != request.initial_answer,
            status=EnhancementStatus.SUCCESS,
            metadata={"adapter": "retrieval"},
        )

executor.register(ControllerAction.RETRIEVAL_ENHANCEMENT, RetrievalEnhancementAdapter(team))
```

### Evidence refinement (`evidence_refinement`)

```python
class EvidenceRefinementAdapter:
    def __init__(self, team_refiner):
        self.team = team_refiner

    def enhance(self, request: EnhancementRequest) -> EnhancementResult:
        answer = self.team(request_to_diagnosis_input(request), request.controller_decision)
        return EnhancementResult(
            action=request.controller_decision.action,
            final_answer=answer,
            evidence=request.evidence,  # or team-refined evidence if exposed
            changed_answer=answer != request.initial_answer,
            status=EnhancementStatus.SUCCESS,
            metadata={"adapter": "evidence_refinement"},
        )

executor.register(ControllerAction.EVIDENCE_REFINEMENT, EvidenceRefinementAdapter(team))
```

### Generation repair (`generation_repair`)

```python
class GenerationRepairAdapter:
    def __init__(self, team_repair):
        self.team = team_repair

    def enhance(self, request: EnhancementRequest) -> EnhancementResult:
        repaired = self.team.generation_repair(
            request_to_diagnosis_input(request), request.controller_decision
        )
        return EnhancementResult(
            action=request.controller_decision.action,
            final_answer=repaired.final_answer,
            evidence=request.evidence,
            changed_answer=repaired.final_answer != request.initial_answer,
            status=EnhancementStatus.SUCCESS,
            metadata={"adapter": "generation_repair"},
        )

executor.register(ControllerAction.GENERATION_REPAIR, GenerationRepairAdapter(team))
```

## Host startup sketch

```python
from src.pipeline import build_adaptive_pipeline, run_adaptive_rag
from src.compensation import EnhancementExecutor
from src.controller.schema import ControllerAction

executor = EnhancementExecutor()
executor.register(ControllerAction.RETRIEVAL_ENHANCEMENT, retrieval_adapter)
executor.register(ControllerAction.EVIDENCE_REFINEMENT, evidence_adapter)
executor.register(ControllerAction.GENERATION_REPAIR, generation_adapter)

pipeline = build_adaptive_pipeline(
    reference_rag=reference_rag,          # injected, loaded once
    classifier=classifier,                # loaded once
    selection_adapter=selection_adapter,  # frozen P60 artifacts
    feature_provider=feature_provider,    # loaded once
    executor=executor,
)
result = run_adaptive_rag("who wrote ...", pipeline=pipeline)
```

## Fallbacks

| Case | Behaviour |
|---|---|
| `action=no_action` | keep `initial_answer` + evidence; pipeline `status=no_action` |
| action needs Enhancement but none registered | keep initial; pipeline `status=enhancement_unavailable` |
| implementation raises | keep initial; pipeline `status=enhancement_error` |

The host pipeline must not crash when a lane is missing.

Mocks for contract tests live in `src/compensation/mocks.py` and are **not**
production defaults.
