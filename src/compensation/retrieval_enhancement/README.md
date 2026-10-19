# Frozen NQ P3-20 entry point

`P320RetrievalEnhancer` uses original-query BGE Top-20, 20 cross-encoder scores,
equal-weight BGE/CE ordinal RRF (`k=60`), and Top-5 context. CE ties and exact
RRF ties break by BGE position. No query rewriting, text deduplication, Probe
hints, score thresholds, retries or additional rounds are applied. The older
`RetrievalEnhancer.enhance()` and rule rewriter remain available as legacy
explicit APIs; they are not the default frozen Controller integration.

## Dependency injection

```python
from src.compensation.retrieval_enhancement import NQFaissBackend, P320RetrievalEnhancer

# All resources are loaded and owned by the external runner, never this module.
# scorer(query, texts) returns 20 finite raw logits, in input order.
# scorer.model_name = "BAAI/bge-reranker-large"
# scorer.revision = "55611d7bca2a7133960a6d3b71e083071bbfc312"
# Owner must load that revision, CPU float32, max_length=512, Identity activation
# to reproduce Phase 5-6; declarations alone do not verify the loaded weights.
# generate_fn(prompt) returns a string using frozen C1 Qwen/NF4/greedy settings.
pipeline = P320RetrievalEnhancer(
    NQFaissBackend(loaded_bge_retriever, max_top_k=20), scorer, generate_fn,
)
retrieval_enhancement = pipeline.retrieval_enhancement
final_answer = retrieval_enhancement(item, decision)  # exactly two arguments
record = pipeline.run(item, decision)  # alternatively; do not call both per sample
```

The module never loads models, imports Torch/Transformers/FAISS, or owns their
lifecycle. Tests use fake providers. No generator inference was validated in
Phase 6-0. `build_c1_prompt` is byte-for-byte tested against the actual frozen
NQ builder via its pure AST, without importing the model-loading module.

## Frozen structural input contracts

Verified local refs at implementation time:

| Ref | SHA | Frozen schema files present? |
| --- | --- | --- |
| feature/retrieval-enhancement (parent) | 2f90bd8512a54038310f1cde53441263cb2f8a75 | No |
| main | c15977cb8ebbaa16a0ada809f72bd57d0893b120 | No |
| origin/main | 058f10baa47083220f95edb39db8f106aa7e85d6 | No |
| origin/reranking | 338ff34b039ed367220b0b916b48c4f7fc715e20 | Yes |

The frozen definitions are `src/controller/schema.py` (`ControllerDecision`,
`ControllerAction`) and `src/diagnosis/schema.py` (`DiagnosisInput`,
`EvidenceItem`), documented by `docs/DIAGNOSIS_PROBE_CONTROLLER_HANDOFF.md`
on origin/reranking. No branch is merged or copied. This implementation accepts
structurally compatible dataclasses or mappings:

- item: `sample_id` (string), `dataset`, `query`, ordered `evidence` (exactly five
  records with `passage_id`, `text`, `score`), and observable `initial_answer`.
- decision: reads only `action`; `ControllerAction` string-enum values work.
  Diagnosis labels, confidence and Probe metadata are not consulted.

No full-object serialization or dictionary copying is used on external inputs.
Gold/reference answers, short_answer, target labels, EM, correctness and offline
evaluation fields are never read. `initial_answer` is the runtime C1 prediction,
required solely for fallback; it is not used in retrieval, scoring or prompts.
Inputs are not modified. Conflicting text for an ID, including Unicode escape
representation differences, fails explicitly; no automatic decoding occurs.

## Outcomes and accounting

- `not_routed`: zero retrieval/scoring/generation calls, keep original answer.
- `no_change`: identical ordered IDs/texts (rounded retrieval scores do not
  affect the prompt), no generation.
- `answer_replaced`: one generated answer passes nonempty, refusal and grounding
  checks. Grounding requires a contiguous normalized token span in one passage.
- `kept_original`: generated answer fails a check; retain original prediction.
- `unsupported_dataset`: routed non-NQ input, no provider calls.
- `<stage>_failed`: provider or result validation failure; retain original answer
  and report the stage/type/message. No retries.

`run()` returns original evidence, all 20 candidates, selected evidence, all CE
scores/ranks and exact RRF sums, initial/generated/final answers, prompt, status,
rejection/error, and per-stage/total wall times. Raw retrieval score is preserved.
`retrieval_calls`, `scoring_calls`, `generation_calls` count attempted provider
API invocations (including exceptions), not internal model forward batches.
`scored_pairs` counts validated returned pair scores. `llm_calls` equals
generation calls; no rewrite/Diagnosis/Probe calls occur here. External runner
must separately record provider batch/forward counts if those are needed.

## Zero-inference saved-score verification

```sh
python3 -B scripts/check_nq_p320_saved_scores.py \
  --scores /path/to/phase5_6/crossencoder_scores.jsonl \
  --selections /path/to/phase5_6/evaluation/selection_simulations.jsonl \
  --output-dir /path/to/a/new/output_directory
python3 -B -m unittest discover -s tests -v
git diff --check
```

The checker requires a new output directory, validates journal SHA-256 records,
full reranker revision, unique sample IDs/count/order, and ordered Top-5 against
stored P3-20 selections. It does not open Gold/answer/Diagnosis datasets. It
writes selections, source checksums and an explicit pass/fail report; incomplete
or mismatching results are errors, never successful consistency claims.
