# Production Adaptive RAG · controller_2

The official production pipeline is frozen. The algorithm, thresholds, the P60 78-dimensional definition, Diagnosis and the Controller mapping will no longer be changed.

Registration file: [current_baseline.json](../configs/controller/current_baseline.json)  
Baseline name: `diagnosis_20261024_p60_p64_v1`  
Reference RAG: [reference_rag_v1.json](../configs/controller/reference_rag_v1.json)

## Architecture

```text
raw Query
  → Reference RAG (matched NQ-Train BGE Top-5 + context_fallback + Qwen2.5-7B)
  → Diagnosis
  → P60 78D online FeatureProvider
  → Frozen Selection
  → Controller
  → EnhancementExecutor
  → {no_action | retrieval_enhancement | evidence_refinement | generation_repair}
  → AdaptiveRAGResult / Final Answer
```

Probe is turned off per the official design (`probe_enabled=false`).

## Reference RAG

| Item | Official configuration |
|---|---|
| Embedding | `BAAI/bge-base-en-v1.5` |
| Index / corpus | matched NQ-Train (`nq_train_bge_base*`, about 1,224,119 passages) |
| Top-k | 5 |
| Generator | `Qwen/Qwen2.5-7B-Instruct`, NF4, deterministic generation |
| Prompt | `context_fallback` (`FREEZE_CONTEXT_FALLBACK`) |

The old Dev index (`nq_bge_base`) is forbidden for production; it is rejected at load time.

## Diagnosis / Selection

- **Diagnosis**: registered classifier `models/diagnosis/diagnosis_v2_expanded_hierarchical.json`
- **P60 FeatureProvider**: `src.controller.p60_online_features.build_p60_online_feature_provider`, 78 dimensions, including the P26 semantic runtime; `metered_runtime=true`
- **Frozen Selection**: `configs/controller/p60_online_adapter.json` + three HGB joblibs; thresholds and z-stats frozen
- **Provenance**: `artifacts/controller/p60_online_adapter/training_provenance.json`

## Controller actions

| Diagnosis / selection | Action |
|---|---|
| `selected=False` | `no_action` (final answer = initial answer) |
| `evidence_insufficiency` | `retrieval_enhancement` |
| `evidence_accessibility` | `evidence_refinement` |
| `evidence_utilisation` | `generation_repair` |

`E_mode` (NQ/FEVER) is officially `allow_repair` (`evidence_id_v2a`).

## Official entry point (raw query)

```python
from pathlib import Path
from src.pipeline.current import load_current_pipeline

root = Path(".")
pipeline, env = load_current_pipeline(
    root=root,
    nq_resources=root / "configs/controller/nq_train_resources_v1.json",
    data_role="validation",  # or "train"; test forbidden
    attach_reference_rag=True,
)
result = pipeline.run("your question here")
print(result.final_answer)
```

Readiness check:

```bash
python scripts/check_controller_current.py
```

It should report `ready_for_gpu=true`. Do not treat the engineering smoke helper as the official API.

## Runtime assets (not in Git)

The following resources must exist locally; their paths are registered in config / registry, and the files themselves are excluded by `.gitignore`:

- matched NQ-Train FAISS index / ids / offsets / corpus (`data/processed/...`)
- Diagnosis and P60 HGB model weights (`models/...`)
- Hugging Face / local cache: `BAAI/bge-base-en-v1.5`, `BAAI/bge-reranker-large`, `BAAI/bge-reranker-base`, `Qwen/Qwen2.5-7B-Instruct`
- P26 semantic runtime arrays (if needed by online features)

Small official config, provenance and registry files **should** go into Git.

## Verification status

Production E2E smoke (Train raw query, full pipeline) has covered:

- `no_action`
- `retrieval_enhancement`
- `evidence_refinement`
- `generation_repair`

No mock / unavailable / missing implementation / exception.  
Evidence summary: `artifacts/controller/production_e2e_smoke_v1/smoke_results_final.json`.

This smoke only proves that the engineering path runs; it is **not** an accuracy or gain conclusion. The frozen Test (11,114) was not run in this stage.
