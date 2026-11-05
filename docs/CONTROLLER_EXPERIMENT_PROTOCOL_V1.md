# Controller experiment protocol v1 · T1

Date: 2026-10-23. Branch: `controller`. Reference implementation: `86a7b126225f7cbc5ffe4c7af7db3137acbec9f5`.

**T1 freezes the comparison design and does not claim that the conditions for a formal run are already in place.** The model, existing components, comparison groups, budget, scoring and stopping rules are fixed here; T2 seals the samples and resources, and after T3 completes the comparison groups and run acceptance, a separate execution checklist is created. There are currently no new GPU experiments or accuracy conclusions.

The machine-readable protocol is in [experiment_protocol_v1.json](../configs/controller/experiment_protocol_v1.json); SHA-256s of the original implementation, prompts, scoring and classifiers are in the [reference snapshot](../configs/controller/protocol_v1_reference_sources.json). The snapshot includes dependencies and historical policy source code, which does not mean all those policies are enabled. The original frozen E experiment and its conclusions are not reopened or overwritten.

## 1. Questions answered in this round

1. On the same batch of C1 inputs, is Diagnosis → Probe → Controller → enhancement more accurate than keeping C1? How many are fixed, how many are broken, and how much extra does it cost?
2. Is dynamic selection better than always invoking one particular enhancement? Relative to Full Enhancement, can it maintain close quality with fewer calls?
3. If there is no net improvement, does the loss come from wrong routing, the enhancement modules themselves, or the interaction of the two?

The main experiment fixes the generation model: Qwen is not trained, and distillation, DPO, knowledge-base cleaning, automatic prompt search or RSI are not introduced at the same time. These can be independent extensions and need their own versions and comparison groups.

## 2. Fixed components and explicit pending items

| Item | Agreement for this round |
|---|---|
| Generation model | Both datasets share `Qwen/Qwen2.5-7B-Instruct`; revision/tokenizer `a09a35458c702b33eeacc393d103063234e8bc28`; NF4, FP16 compute, double quant; greedy, seed 42 |
| C1 | For the same question, every group reads exactly the same cached answer and ordered Top5; the original C1 model/prompt/index provenance needs T2 verification, and an unknown revision cannot be inferred backwards or filled in |
| Retrieval | `BAAI/bge-base-en-v1.5`; original query; the corpus/index/metadata matching this batch of C1; the old dev index cannot substitute for the NQ Train index |
| Retrieval enhancement | NQ P3-20: Top20 + reranker, equal-weight rank RRF, k=60, back to Top5; query unchanged, no additional retrieval loop |
| Reranker | `BAAI/bge-reranker-large`; revision `55611d7bca2a7133960a6d3b71e083071bbfc312`; CPU/FP32/raw logits, max_length 512 |
| Reranking filter | Original Top5, keep5, no score threshold, probe_hint_weight=0, require_grounded=true; no re-answer if the order is unchanged |
| Diagnosis | `diagnosis_v2_expanded_hierarchical.json` original parameters and 27 features; single dominant type, no claim of multi-label diagnosis |
| Probe | Original prompt; trigger interval [0.50, 0.65], otherwise triggered only when stage2 exists and the maximum score is <0.50; at most once |
| Safe Binary | Original rules; only safe overrides of failure → correct are allowed, thresholds stage1 ≤0.70, stage2 max ≤0.50; no switching among the three failure types |
| E / NQ | `short_answer_quote`, prompt version `v7_short_answer_in_quote_lines`; Qwen generates the short answer and quote, and the program checks format/source |
| E / FEVER | `verify_repair`, prompt version `v3_balanced_examples`; the same Qwen verifies and outputs the label/quote; no independent NLI is called in this round |

This E configuration is the **integration comparison baseline**; it does not represent a reproduction of the historical v7 FEVER NLI, nor has it been declared the winning strategy. The current source's `conservative_repair` is r4; this strategy name cannot be used to claim that the old r3 was run.

BGE's actual weight revision, the resource hashes of both datasets, the original C1 provenance, hardware and the full dependency lock still need T2/T3 sealing. Merely writing the model name, having the same number of passages, or passing structural checks cannot prove the resources match. If the old C1 source cannot be verified, one must choose and record before the first comparison: keep it as a "historical C1 post-processing experiment", or rebuild a traceable C1; they cannot be quietly mixed.

## 3. Six main comparison groups and two explanatory controls

The groups below use the same questions, C1, gold, model and component policies. `always` means the action is attempted on every question; the module's own rejection/fallback rules are still kept.

| Mode | Behaviour | Current status |
|---|---|---|
| `c1` | Keep the original answer | Exists |
| `always_retrieve` | Attempt retrieval enhancement and re-answer on every question | Exists for NQ; FEVER pending T3 |
| `always_rerank` | Attempt reranking on every question and re-answer per the original rules | Exists |
| `always_repair` | Attempt E on every question | Exists |
| `full_enhancement` | Fixed order: retrieval enhancement → reranking → E; no diagnosis, no Probe | Pending T3 |
| `adaptive` | Diagnosis → optional Probe → Safe Binary → one action or keep | Exists; FEVER retrieval enhancement pending T3 |
| `adaptive_no_probe` | Same Diagnosis, Probe and its overrides turned off, everything else unchanged | Explanatory ablation, pending T3 |
| `human_type_routed` | Offline human dominant type determines the action; correct is kept; unknown is listed separately and not executed | Explanatory control, pending T3; uses only the common subset with reliable type annotation |

Human types can only be converted into actions in the offline comparison driver; gold answers, gold evidence, correctness or human rationales cannot be put into model prompts. It measures the effect "given the human type", and is **not necessarily the upper bound of the optimal action**. The failure type × three actions matrix directly reuses the fixed-action outputs; when aggregating by type, the same reliably annotated subset must be used and the sample count stated.

Full's state passing is fixed as the `final_answer` and `final_evidence` accepted by the previous stage. When the answer is unchanged, the current executor's rule for keeping context is followed; rejected candidates/contexts do not enter the next stage. If a stage fails, the state entering that stage is kept, the error recorded, and the remaining stages are still attempted within budget; the shared budget is never reset. The original query is unchanged, with no free model reranking step or extra retries. The specific definition of this Full must be reported with the results, rather than generically referring to every possible "full enhancement".

Currently FEVER retrieval returns `unsupported_dataset`, which cannot count as implemented. The formal comparison across both datasets needs T3 to complete it. If resources are insufficient, it can only be sealed before seeing results as "NQ-only full comparison, FEVER partial path"; FEVER cannot be dropped after seeing gains, nor can fallback be treated as Full.

T3's FEVER retrieval enhancement is completed with the same algorithm: the original claim takes Top20 through the BGE/FAISS backend matching C1, the same reranker and RRF select Top5, then the original FEVER C1 prompt is used to re-answer; only parseable SUPPORTS/REFUTES are accepted, otherwise the answer entering the stage is kept. NQ's answer span check cannot be applied directly to FEVER labels; this difference must be disclosed in the implementation and results; a parseable label still does not equal semantic correctness. The two explanatory controls are first used for development/selection-run diagnosis, and the final held-out set runs the six main groups in a fixed manner.

## 4. Call and output budget

The budget starts **after** the cached C1, and includes Probe. Every provider call is counted, including calls that raise exceptions; one scoring means scoring one candidate list, not one batch inside a model.

| Mode | Qwen call cap | Retrieval enhancement calls | Scoring calls | Cap on sum of max_new_tokens across stages |
|---|---:|---:|---:|---:|
| C1 | 0 | 0 | 0 | 0 |
| Always Retrieve | 1 | 1 | 1 | 64 |
| Always Rerank | 1 | 0 | 1 | 64 |
| Always Repair | 1 | 0 | 0 | 384 |
| Full | 3 | 1 | 2 | 512 |
| Adaptive | 2 | 1 | 1 | 544 |
| No Probe / Human Type | 1 | 1 | 1 | 384 |

These are worst-case path caps for the selected components, not the amount necessarily consumed per question; the column maxima of mutually exclusive actions do not necessarily occur on the same path. Probe=160, retrieval-enhancement re-answer=64, reranking re-answer=64, E=384. Input context follows the corresponding template and the existing truncation/over-length rejection contract; evidence length or output budget is not increased separately to make some group look good.

The current generic hard caps in `integration_v1.json` are Qwen 4, retrieval 1, scoring 1. They accommodate the current single route but are not enough for Full's two scorings. **T3 generates an explicit budget configuration for each group per the table above and validates execution traces**; not all groups are changed to an unlimited budget, and the modules' original prompts are not modified. Differences in actual consumption are one of the comparison targets; empty calls are not artificially padded.

Same-batch comparisons fix hardware, precision, dependencies and resources, and record arm execution order and warm-up; load time is listed separately. Tokens of failed calls are recorded as null when unknown. Partial token sums must be marked as partial observations; missing values cannot be treated as zero, and a complete cost-saving ratio cannot be claimed on that basis. Besides incremental cost, if C1 cost is traceable, the common C1 cost is added to report full-pipeline cost; the incremental cost-saving ratio cannot be passed off as the full-pipeline ratio.

## 5. Data, sample size and isolation

T2 submits the data manifest before any new model output is produced: data source, filtering code/rules, seed 42, candidate population, N for each dataset/split, ID lists, exclusion reasons, input/gold SHA-256, exposure registration, resource mapping and the estimation precision corresponding to the sample size. **This T1 does not fabricate a frozen sample size, nor treat the 74,097 items as a used test set.**

- The main comparison samples from the population meeting pre-specified data-integrity conditions, without filtering questions by EM=0, a particular failure type, expected repairability or human preference; correct original answers must be included in order to estimate harmful repair. Failure-type-enriched sets are reported separately and cannot represent population accuracy.
- NQ uses the original data's reference short answers; FEVER's scope in this round is the SUPPORTS/REFUTES subset with qualified sources, and this score is not generalised to the complete FEVER including NEI.
- Keep development, selection-run and final held-out data distinct. Development may diagnose errors; the final held-out set is run only once after the final version is frozen. NQ's already-used 75/125, E-exposed IDs, Diagnosis training/validation and already-viewed tests are all registered; unexposed to E does not equal unexposed to the team.
- Newly delivered AI/heuristic diagnosis labels cannot be treated as full human gold. They can help troubleshooting, but the main QA scoring still comes from official answers/labels, and the human-type control has separate provenance.
- Handle cross-split duplicates and known issues first and record the rules; published splits and results remain as is. Exclusions may only rest on prior structural/source conditions; questions cannot be deleted after seeing the model answer wrongly.
- After freezing, gold aliases must not be modified to give new answers extra points. Reasonable semantic-equivalence/incompleteness disputes are saved as independent human review records, while the original automatic scores are retained.

## 6. Scoring, cost and conclusion boundaries

Primary metrics: NQ normalized EM; FEVER label accuracy. NQ token F1 is auxiliary. The two datasets are reported separately; EM and label accuracy are not mixed into an undefined average. Formulas:

```text
fixed count = questions with original answer wrong and final answer right
harmful repair count = questions with original answer right and final answer wrong
net correct gain = fixed count - harmful repair count
accuracy difference = net correct gain / all paired questions
fix rate = fixed count / questions with original answer wrong
harmful repair rate = harmful repair count / questions with original answer right
```

A zero denominator is recorded as N/A. Format errors, no-ops, rejections, budget overruns and provider failures are all kept and listed separately; computation cannot be restricted to "successfully executed" samples. Per dataset, every group is checked to match the frozen ID list exactly, and all paired questions use the same C1 input hash. A group that is entirely missing cannot pass formal acceptance just because the remaining groups pair with each other.

Statistics agreed in advance: the accuracy difference uses a per-question paired bootstrap, 10,000 iterations, seed 42, reporting the 95% percentile interval; exact two-sided McNemar test. Confirmatory comparisons are limited to final adaptive vs C1 on the two datasets, with Holm correction of the two p-values; other groups and type subsets are explanatory comparisons. Numbers on the development set only guide the selection run and are not independent confirmatory evidence. The existing evaluator has McNemar, but bootstrap, Holm and acceptance against the complete frozen list still need T3 to complete them.

Human review prioritises covering **the union of all changed answers** in the main groups and key ambiguities, hiding method names, shuffling the positions of the two answers and fixing the grading criteria; when blind review is impossible, the influence of assisted review is disclosed. Keep "correct/incorrect/pending" and the rationale. Reviewing only changed answers can estimate the reviewed net change and the pending upper/lower bounds, but cannot be used to report overall semantic accuracy, nor to use the correctness of unreviewed C1 as the denominator of the complete fix rate/harmful repair rate. A quote/short answer appearing in the evidence only shows that the source check passed, not semantic support.

Cost reporting includes at least actual Qwen/Probe, retrieval and scoring counts, token observation coverage, incremental end-to-end latency, load time, failure counts, route distribution, no_action proportion and unsupported proportion. Costs are compared only with complete records, the same environment and complete pairing. Human/large-model offline review costs are listed separately and cannot be mixed into online model quality scores.

## 7. Continuation, stopping and version discipline

- T4 first completes one frozen baseline comparison. T5 uses the same batch of fixed-action outputs, No Probe and the reliable human-type subset to locate the largest loss; it is not presumed that all downstream problems are caused by Diagnosis.
- After the baseline, at most one round of focused optimisation: pre-register one candidate system for the largest bottleneck, with its changes, rationale and version hash; if the candidate involves multiple components at once, attribution limits must be stated, and the whole-package gain cannot be allocated to a single module. Endless prompt search and repeatedly using old smoke runs to tune semantics are not in the plan. Necessary implementation bug fixes are recorded separately, and the related paired groups are re-run uniformly.
- On the selection-run data frozen in T2, a candidate is promoted only if its net correct gain over the baseline is >0; on a tie the baseline is kept; on a regression the baseline is kept or the harmful action is disabled. The final held-out set must not take part in the selection run. If there is no credible net gain, report the scope of applicability/negative result truthfully, without forcing a positive number by adding data, changing scoring or continuing to loop.
- The final held-out set is run once, reporting C1/final adaptive and the frozen main comparison groups. A better sample point estimate is called "an observed improvement"; claiming statistical evidence requires support from intervals and corrected tests. No significant difference on a small sample does not prove ineffectiveness.
- The Proposal's "token reduction ≥20% relative to Full, accuracy drop ≤2 percentage points" is kept as an engineering target; it has not yet been reached and is not a guarantee. Incremental/complete costs are reported separately; a point estimate meeting the target does not equal statistical non-inferiority, which needs corresponding uncertainty analysis for support.
- Semantically wrong answers and parse failures are not retried. A definite infrastructure interruption may be restarted at most once with the same input/code/model/environment, keeping the failure record and creating a new run directory; the best answer per question is not cherry-picked. When an effective computation version changes, a new list is created and the whole group pairing is re-run.

T1's original protocol and reference snapshot stay unchanged. T2/T3 generate an independent data manifest and execution seal, referencing T1 hashes and registering differences relative to the reference implementation; completing engineering capabilities does not mean existing prompts/thresholds can be re-tuned. Changing the model, scoring, main comparison groups, filtering or budget requires writing a protocol revision version before the run; revisions after the run can only be new experiments.

## 8. Deliverables and start-up gates

| Task | Verifiable results that must be produced |
|---|---|
| T1 (this round) | This document, the machine protocol, the reference SHA list, a read-only checker, regression tests and the commit |
| T2 | Paired inputs/gold, IDs and N, team exposure check, splits, index/metadata/BGE revision and C1 provenance; no training labels mixed into runtime |
| T3 | Full/FEVER retrieval/No Probe/human-type driver, per-group budget and coverage, statistics and list acceptance, CUDA environment lock, execution seal |
| Before T4 starts | Verify real CUDA, model/index, call metering and output on fictitious/already-exposed engineering samples; do not rely on unseen questions to tune formatting |

Check T1 (read-only, no model loading, no reading of questions or gold):

```bash
python scripts/check_controller_protocol.py
python scripts/check_controller_protocol.py --require-execution-ready
```

The first returns 0 when the protocol/reference snapshot is self-consistent; workspace changes relative to the reference implementation are listed separately and not automatically accepted as a new baseline. The second explicitly returns 2 in this design stage and lists the T2/T3 gates. **The T1 checker does not approve GPU runs, nor has it taken over the existing runners**; formal start-up acceptance is the responsibility of T3's execution seal check.
