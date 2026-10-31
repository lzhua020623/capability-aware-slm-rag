# Production Adaptive RAG · controller_2

本页记录历史 controller_2 四动作基线；当前默认主线已切换到 [B + BM25 + 自适应答案选择](../configs/controller/README.md)。
复现本页时须显式指定 `configs/controller/current_baseline.json`。

以下历史生产链路已冻结。算法、阈值、P60 78 维定义、Diagnosis 与 Controller mapping 不再改动。

登记文件：[current_baseline.json](../configs/controller/current_baseline.json)  
基线名：`diagnosis_20261024_p60_p64_v1`  
Reference RAG：[reference_rag_v1.json](../configs/controller/reference_rag_v1.json)

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

Probe 按正式设计关闭（`probe_enabled=false`）。

## Reference RAG

| 项 | 正式配置 |
|---|---|
| Embedding | `BAAI/bge-base-en-v1.5` |
| Index / corpus | matched NQ-Train（`nq_train_bge_base*`，约 1,224,119 passages） |
| Top-k | 5 |
| Generator | `Qwen/Qwen2.5-7B-Instruct`，NF4，确定性生成 |
| Prompt | `context_fallback`（`FREEZE_CONTEXT_FALLBACK`） |

旧 Dev index（`nq_bge_base`）禁止用于 production；加载时会拒绝。

## Diagnosis / Selection

- **Diagnosis**：登记分类器 `models/diagnosis/diagnosis_v2_expanded_hierarchical.json`
- **P60 FeatureProvider**：`src.controller.p60_online_features.build_p60_online_feature_provider`，78 维，含 P26 semantic runtime；`metered_runtime=true`
- **Frozen Selection**：`configs/controller/p60_online_adapter.json` + 三个 HGB joblib；阈值与 z-stats 冻结
- **Provenance**：`artifacts/controller/p60_online_adapter/training_provenance.json`

## Controller actions

| Diagnosis / selection | Action |
|---|---|
| `selected=False` | `no_action`（终答 = 初答） |
| `evidence_insufficiency` | `retrieval_enhancement` |
| `evidence_accessibility` | `evidence_refinement` |
| `evidence_utilisation` | `generation_repair` |

`E_mode`（NQ/FEVER）正式为 `allow_repair`（`evidence_id_v2a`）。

## 正式入口（raw query）

```python
from pathlib import Path
from src.pipeline.current import load_current_pipeline

root = Path(".")
pipeline, env = load_current_pipeline(
    registration="configs/controller/current_baseline.json",
    root=root,
    nq_resources=root / "configs/controller/nq_train_resources_v1.json",
    data_role="validation",  # or "train"; test 禁止
    attach_reference_rag=True,
)
result = pipeline.run("your question here")
print(result.final_answer)
```

就绪检查：

```bash
python scripts/check_controller_current.py --registration configs/controller/current_baseline.json
```

应报告 `ready_for_gpu=true`。不要把工程 smoke helper 当作正式 API。

## Runtime assets（不进 Git）

下列资源本地必须存在；路径登记在 config / registry，文件本身由 `.gitignore` 排除：

- matched NQ-Train FAISS index / ids / offsets / corpus（`data/processed/...`）
- Diagnosis 与 P60 HGB 模型权重（`models/...`）
- Hugging Face / 本地缓存：`BAAI/bge-base-en-v1.5`、`BAAI/bge-reranker-large`、`BAAI/bge-reranker-base`、`Qwen/Qwen2.5-7B-Instruct`
- P26 semantic runtime 数组（若在线特征需要）

小型正式 config、provenance 与 registry **应**进入 Git。

## 验证状态

Production E2E smoke（Train 裸 query，完整链路）已覆盖：

- `no_action`
- `retrieval_enhancement`
- `evidence_refinement`
- `generation_repair`

无 mock / unavailable / missing implementation / exception。  
证据摘要：`artifacts/controller/production_e2e_smoke_v1/smoke_results_final.json`。

该 smoke 只证明工程路径可运行，**不是**准确率或收益结论。冻结 Test（11,114）未在本阶段执行。
