# Evidence Refinement by Reranking

Owner: the `evidence_refinement` Enhancement module, the action the frozen
Controller routes `evidence_accessibility` to.

`evidence_accessibility` means the needed information **is** in the retrieved
evidence, but noise, conflict or ordering stops Qwen-A from using it
(`docs/DIAGNOSIS_PROBE_INTERFACE.md`). So this module does not retrieve
differently: it reranks what was retrieved with a cross-encoder, prunes the
noise, re-asks the frozen generator once, and keeps the original answer unless
the new one is grounded in the evidence that survived.

## 1. Frozen baseline this module is measured against

Nothing below may be changed by this module. All three freezes are documented
in the repository, not tagged in git.

| Layer | Frozen content | Source |
|---|---|---|
| Retriever | `BAAI/bge-base-en-v1.5`, L2 + inner product, Top-5, chunk 512/64 | `configs/base.yaml`, `docs/RETRIEVAL_CONFIG.md` |
| Protocol | sample IDs, prompts, decoding, metrics, C0/C1/C3 | `configs/preliminary.yaml`, `docs/PRELIMINARY_EXPERIMENT.md` |
| Diagnosis + Probe + Controller | classifier, probe prompt, trigger, Safe Binary, routing | `docs/DIAGNOSIS_PROBE_CONTROLLER_HANDOFF.md` |

Reference RAG C1: `Qwen/Qwen2.5-7B-Instruct` 4-bit NF4, `do_sample=False`,
`max_new_tokens=64`; NQ `IndexFlatIP` over 99,922 passages; FEVER
`IndexIVFPQ` over 5,396,106; seed 42.

**Retrieval pilot** (NQ, 3,216 queries): BGE Recall@3 0.6872, Recall@5
0.7718, Recall@10 0.8451. (E5-base-v2 0.6894 / 0.7705 / 0.8567, comparison
only.)

**Generation, 500 frozen samples per dataset:**

| | C0 no evidence | C1 Top-5 | C3 gold evidence |
|---|---|---|---|
| NQ EM | 0.102 | **0.352** | 0.516 |
| NQ token-F1 | 0.194 | **0.490** | 0.678 |
| FEVER accuracy | — | **0.804** | 0.952 |

Recoverable Failure Rate: NQ 29.9% (97 of 324 C1-wrong), FEVER 81.6% (80 of
98). Of the 7B's C1+C3 double failures a Llama-3.3-70B under C3 recovers NQ
15/67 and FEVER 8/18, so 62 samples are unresolved hard cases.

**C1 broken down by annotation label** (frozen 500 NQ C1 records joined with
`data/annotations/normalized/nq_3216_labels_normalized.jsonl`):

| Annotation label | n | EM | token-F1 | gold evidence in Top-5 |
|---|---|---|---|---|
| `correct` | 375 | 0.469 | 0.638 | 322/375 |
| `evidence_insufficiency` | 54 | 0.000 | 0.016 | 1/54 |
| **`evidence_accessibility`** | **38** | **0.000** | **0.075** | **23/38** |
| `evidence_utilisation` | 33 | 0.000 | 0.058 | 30/33 |

The accessibility row is this module's starting point: EM 0, token-F1 0.075.
Two caveats: `gold_evidence_in_top5` is a gold-passage-ID match while the
annotators judged semantic presence, so the 23/38 understates how often the
information is actually there; and 21 of those 38 carry
`needs_manual_review=true`, making accessibility the noisiest of the four
classes.

**Label population.** 3,216 annotated NQ samples hold 116 accessibility (68
trusted). In the frozen `diagnosis_v2_expanded` split the trusted
accessibility samples are 47 train / 11 validation / 10 test. FEVER has none.

**What actually reaches this module** (frozen PRIMARY test, n = 473):
`evidence_refinement` is chosen for 29 samples, whose true labels are 15
`correct`, 10 `insufficiency`, **1 `accessibility`**, 3 `utilisation`.
Accessibility per-class precision 0.034, recall 0.100, F1 0.051. So misrouting
is the normal case, and harm avoidance matters more than recall.

## 2. Interface

Frozen contract (handoff section G): the module consumes `DiagnosisInput` and
`ControllerDecision` and returns the final answer.

```python
from src.compensation.evidence_refinement import EvidenceRefiner
from src.controller.controller import run_controller
from src.controller.schema import ControllerAction

refiner = EvidenceRefiner.from_frozen_defaults()      # cross-encoder + Qwen2.5-7B NF4
decision = run_controller(item, classifier, probe)

if decision.action is ControllerAction.EVIDENCE_REFINEMENT:
    final_answer = refiner(item, decision)            # the frozen two-argument call
```

| Entry point | Returns | Use |
|---|---|---|
| `refiner(item, decision)` | `str` | the frozen integration call |
| `evidence_refinement(item, decision, refiner)` | `str` | same, with the refiner passed explicitly |
| `refiner.run(item, decision, probe_result=None)` | `RefinementResult` | full record for logging and offline evaluation |
| `rerank_evidence(query, candidates, scorer, config, probe_result)` | `RerankOutcome` | reranking alone, no generation |

`EvidenceRefiner(scorer, generate_fn, config, candidate_provider, prompt_builder)`
takes every collaborator by injection, so tests load no model. The optional
`candidate_provider(item) -> Iterable[EvidenceItem]` widens the candidate
pool (for example a deeper Top-50 from the frozen index on a machine that has
it); without it the pool is exactly the evidence the generator already saw.
Extra candidates are appended after the original Top-5 and renumbered, so
Probe rank hints keep pointing at the passages the Probe saw.

`RefinementConfig`: `keep_k` (default 5, i.e. reorder and keep everything;
capped at 5), `min_score` (default `None`), `probe_hint_weight` (default 0.0),
`require_grounded` (default `True`), `regenerate_when_unchanged` (default
`False`), `reranker_model` (default `BAAI/bge-reranker-large`). These are the
module's own defaults, not a frozen project setting; section 4 explains why
pruning is opt-in.

`RefinementResult` reports `status`, `initial_answer`, `final_answer`,
`answer_changed`, the evidence before/after/dropped, `rerank_scores`,
`candidate_pool_size`, `regenerated`, `used_probe_hint` and `reject_reason`.
`status` is one of:

| Status | Meaning |
|---|---|
| `not_routed` | the action was not `evidence_refinement`; nothing was touched |
| `no_change` | reranking reproduced the original order, so no generation call was spent |
| `kept_original` | the generator ran, the new answer was rejected |
| `answer_replaced` | the new answer passed the acceptance check |

## 3. Policy

1. **Gate.** Any action other than `evidence_refinement` returns
   `not_routed` without calling the scorer or the generator.
2. **Rerank.** Cross-encoder score per candidate, order by score, ties broken
   on the original rank, keep `keep_k`, apply `min_score` if set but never
   drop the best passage. The refined evidence keeps each passage's original
   retrieval score in `EvidenceItem.score`; reranker scores are reported
   separately.
3. **Probe hints are off by default.** `supporting_ranks` back the *initial*
   answer, the one we were routed here to distrust, so a positive
   `probe_hint_weight` promotes the passages that contradict it. Since about
   half the routed samples are already correct, neither direction is safe as a
   default, and the Probe is a hint rather than ground truth.
4. **Re-ask once**, with the frozen C1 prompt rebuilt over the refined
   evidence (`src/generation/qwen.py` builders; the instruction text is never
   edited). Skipped entirely when reranking changed nothing.
5. **Accept only if grounded.** A replacement answer is rejected when it is
   empty or a refusal, when FEVER output does not parse to SUPPORTS/REFUTES,
   or (NQ) when it does not appear as a contiguous normalised token run in a
   kept passage. On rejection the original answer stands.

The module never re-runs Diagnosis, the Probe or Safe Binary, never changes the
retriever or the frozen prompts, and never reads gold answers, gold evidence,
correctness, EM/F1, C0/C3 or larger-model outcomes.
`src/compensation/schema.assert_no_forbidden_fields` is asserted over the
runtime dataclasses in `tests/test_compensation_schema.py`.

## 4. Offline evaluation so far

`scripts/evaluate_rerank_accessibility.py` reranks the frozen 500 NQ C1
records and asks a retrieval-side question only: after reranking, is a passage
containing the gold answer nearer the top, and does it survive pruning? It
loads no generator, so it says nothing about final answer accuracy. Gold
answers are used here strictly after the fact.

`BAAI/bge-reranker-large`, `keep_k=3`, 500 samples, 182 s on an RTX 3060
(`results/compensation/rerank_frozen500_offline_keep3.json`; per-sample rows
for the accessibility subset in
`results/compensation/rerank_accessibility_offline.json`):

| Label | n | has answer-bearing passage | at rank 1 before → after | mean best rank before → after | promoted / demoted | lost to pruning |
|---|---|---|---|---|---|---|
| **`evidence_accessibility`** | 38 | 25 | 10 → 11 | 1.92 → 1.67 | 10 / 6 | 1 |
| `correct` | 375 | 311 | 219 → 243 | 1.52 → 1.22 | 63 / 24 | 13 |
| `evidence_insufficiency` | 54 | 2 | 1 → 1 | 3.00 → 1.50 | 1 / 0 | 0 |
| `evidence_utilisation` | 33 | 27 | 21 → 17 | 1.41 → 1.41 | 5 / 7 | 0 |

On the target class the reranker does move the answer-bearing passage up (mean
best rank 1.92 → 1.67, 10 promoted against 6 demoted), though on only 25
samples that is a small absolute gain. On `utilisation` it is slightly
harmful, as expected for a class whose evidence was already clear. The
`correct` row is the harm row.

### Why pruning is not the default

The same run with `keep_k=5` — reorder only, nothing dropped
(`results/compensation/rerank_frozen500_offline_keep5.json`):

| | `keep_k=3` | `keep_k=5` |
|---|---|---|
| answer-bearing passage at rank 1 (500 samples) | 272 | 272 |
| accessibility at rank 1 | 11 | 11 |
| answer-bearing passage lost | 14 | **0** |
| mean best rank after | 1.27 | 1.38 |

Pruning cannot change which passage ends up at rank 1, so it buys nothing on
these metrics, and it costs 14 samples (13 of them already-correct ones) their
only answer-bearing passage. The apparently better mean best rank under
`keep_k=3` is survivorship: pruning removes exactly the samples whose answer
sat deep in the list, so they drop out of the average instead of improving it.
The default is therefore `keep_k=5` and pruning is opt-in. The one benefit
pruning may still have — a shorter, less noisy prompt for the generator — is
invisible to a retrieval-side metric and has to be decided on the generator
run below.

The grounding gate in step 5 remains the safeguard whenever pruning is turned
on: if the answer-bearing passage was dropped, the regenerated answer will not
be grounded in the kept evidence and the original answer stands.

**Not yet measured:** whether a refined context actually changes the 7B's
answer, and the EM/F1 gain or harm that follows. That needs a machine holding
`Qwen/Qwen2.5-7B-Instruct` and the NQ index; this one has neither.
The measurement to run there, on the routed samples only:

* EM / F1 before and after, split by the true annotation label;
* harm rate on routed samples whose initial answer was already correct;
* `status` distribution and how often the grounding gate rejected a
  replacement;
* generation calls spent (one per sample that was not `no_change`);
* `keep_k=5` against `keep_k=3`, to settle whether a shorter prompt helps
  enough to pay for the dropped passages.

## 5. Files

| Path | Content |
|---|---|
| `src/compensation/schema.py` | `RefinementConfig`, `RefinementResult`, `RefinementStatus`, forbidden-field guard |
| `src/compensation/rerank.py` | `RelevanceScorer`, `CrossEncoderScorer`, `build_candidate_pool`, `rerank_evidence` |
| `src/compensation/evidence_refinement.py` | `EvidenceRefiner`, `evidence_refinement`, grounding and acceptance rules |
| `scripts/evaluate_rerank_accessibility.py` | offline retrieval-side evaluation |
| `tests/test_compensation_schema.py`, `tests/test_compensation_rerank.py`, `tests/test_evidence_refinement.py` | 68 tests, no model loaded except the frozen prompt builders |
