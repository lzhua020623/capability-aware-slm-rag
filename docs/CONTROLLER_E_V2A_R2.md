# E v2a r2: unified verification label definition · registered 2026-10-26

**Status (2026-10-26)**: registered and implemented; the current v2 package comes from compatibility integration commit `e3ddead`, and the global mode is aligned. The old run package from `9d91a40` is kept; see the [r2 run package](../artifacts/controller/e_v2a_interface_r2/README.md). First GPU run: [Kaggle 362285592](https://www.kaggle.com/code/yearo111/controller-e-r2-shared-frozen-20261026?scriptVersionId=362285592), finished and independently audited: 10 of 20 items passed, structural failures 1, semantic failures 9, execution failures 0. Stopped per registration, development calls are 0, globally unavailable; see [raw results and interpretation](../artifacts/controller/e_v2a_interface_r2/kaggle_v1/README.md). The main line no longer schedules r3.

r2 is a **semantic-definition amendment** registered after the r1 GPU results and before implementing and running r2; it is not an interface correction, nor a bug fix. The machine protocol is in [e_v2a_engineering_r2.json](../configs/controller/e_v2a_engineering_r2.json). r1's outputs, audit and `unavailable` decision are kept as is in [e_v2a_interface_r1](../artifacts/controller/e_v2a_interface_r1/README.md); r2 does not overwrite them.

## Why it was changed

On the NQ missing-fact question in r1, the rationale said "the evidence does not give the opening year", yet the label given was `unsupported`, when the expectation was `insufficient`. The original answer was kept and marked; no harmful repair occurred.

Our explanation is that the label definition is asymmetric: the FEVER prompt states "missing or conflicting evidence counts as insufficient", while the NQ prompt lacks this sentence; moreover, the English word unsupported can itself mean "not supported". This is a hypothesis to be tested; a single-question output cannot prove causation.

## What was changed

1. **Both datasets use the same set of definitions.** The object of judgement is always "the answer entering E": for NQ it is the short answer, for FEVER the SUPPORTS/REFUTES label.
   - **supported**: the evidence establishes that the original answer is correct. For FEVER, this means SUPPORTS with the claim established, or REFUTES with the claim explicitly refuted.
   - **contradicted**: the evidence explicitly proves the original answer wrong, i.e. it gives a different value for the same requested fact, or explicitly rules out the original answer.
   - **insufficient**: the evidence can neither establish nor rule out the original answer, covering two cases: only related facts are present and the requested fact is missing; or several evidence passages conflict with each other and the evidence itself does not say which is right. Lack of support does not equal refutation. In a conflict, source name, authority, time and ordering cannot be used to adjudicate, nor can events such as recasting or rebuilding be invented to reconcile them.
2. **The labels output by the model are changed to `supported / contradicted / insufficient`.** The program maps `contradicted` to the internal `unsupported`; downstream states, fallback and audit are unchanged. The model's raw output label is saved in `model_label`.
3. **The NQ replacement answer takes the shortest contiguous original-text span that answers the question**, at most 20 words. When judged wrong but with no qualified replacement, the original answer is kept and marked unresolved; the system never abstains.
4. **A new NQ answer-type check is added, which only records warnings.** It checks the question type against the form of the replacement answer (when needs a time expression, how many needs a number, who/where need a proper noun), writes to `type_warnings`, and does not block, downgrade or change the answer. Automatic protection is left as an extension.

Unchanged parts:

- Model and runtime: Qwen2.5-7B / NF4 / greedy / seed 42, single call, 384 output tokens, XGrammar 0.2.8, prefill256;
- Evidence and output constraints: the shared RULES text, sentence numbering and quote reconstruction, schema branch structure and ID count, NQ's 20-word contiguous span check, FEVER's reverse label;
- Data and evaluation: data and gold, the 800-item development plan, routing search.

The internal version number is `evidence_id_v2a_v3`.

## GPU acceptance

The acceptance questions were written before any prompt was modified, and sealed after three rounds of independent blind review; see the [T5.1 sealing record](../artifacts/controller/e_v2a_interface_r2/fixture_review/README.md):

- 18 short questions: five NQ categories and four FEVER categories, 2 each;
- 2 long questions: 14k–32k tokens, with XGrammar and prefill256 both enabled.

Unlike r1, r2 runs all 20 checks before classifying, rather than stopping at the first failure. Each result is classified in the following order:

| Category | Condition |
|---|---|
| Execution failure | Exception, call failure or blocked by budget, call not completed, output hit the cap; for long questions also input under 14,000 tokens, decoder not xgrammar, prefill not 256 |
| Structural failure | Contract not passed: verification incomplete, illegal source ID, illegal replacement span |
| Semantic failure | Contract passed, but any one of the verification category, final action, whether unresolved, or final answer does not match the sealed expectation |

Key evidence sentences, type warnings and raw labels are only recorded, not used as pass conditions.

If the platform is interrupted before any check completes (no model output at all), one restart with the same experiment package unchanged is allowed and recorded. In all other cases the first run is authoritative.

## Three branches for the development stage

| Acceptance result | Development runs | Notes |
|---|---|---|
| All passed | Always Repair 400 runs + Full's E stage 400 runs, 800 in total | Save the repair-allowed diagnostic output only once; the run mode is derived offline afterwards, without regeneration |
| Only semantic failures | Both datasets run only Always Repair 400 times in verify_only | Full reuses the answers from the first two stages, and E cost is annotated separately as an estimate. This is experimental verify-only, **does not count as E acceptance passed**, and FR12's repair requirement is not met |
| Execution or structural failures present | 0 runs | E for both datasets is set to unavailable, keeping the original answer and marking verification incomplete, written into the report as a limitation |

## Global E mode

NQ, FEVER and all comparison groups containing E share one global mode. The decision is made only offline, without reading runtime gold.

- **unavailable**: acceptance failed; or the development contract compliance rate of either dataset is below 90%. Failures and invalid calls all stay in the denominator.
- **verify_only**: any one of the following conditions holds:
  - the acceptance result is semantic failures only;
  - the union of insufficient and unresolved exceeds 90%: pause and re-examine; changed answers cannot be forced to push the proportion down;
  - the frozen EM net value of NQ's 16 fixed-routed items is negative; FEVER's zero routed samples provide no net value or safety evidence, so the harmful-repair protection condition on its own 200 Always Repair items must be used;
  - the harmful-repair cap is exceeded.
- **allow_repair**: acceptance passed, compliance rate not below 90%, review line not exceeded, NQ fixed-routing subset net value not less than 0, and both datasets satisfy the harmful-repair cap. The net value counts both fixed and broken, and cannot replace the independent harmful-repair protection condition.

**Harmful-repair cap**: the object of the statistic is the diagnostic output on the Always Repair development set:

- FEVER: the number of "originally right, changed to wrong" under frozen labels must be 0.
- NQ: frozen-EM harmful repairs must be 0; all changed answers get semantic review by the user, and confirmed semantic harmful repairs and pending items must also be 0. The two definitions are recorded separately, without changing gold or aliases.

The harmful-repair cap was fixed at 0 before the first r2 GPU run; it is a conservative enabling gate, not a statistical safety proof. The 400-run verify-only development under semantic-failures-only must also check the real contract compliance rate of both datasets; if either is below 90%, it is globally unavailable. A verify-only decision without development statistics is only a temporary run branch, not a final freeze.

## Stopping rules

- The main line no longer schedules r3. Any E semantic amendment after r2 is placed after T8 as an extension.
- Prompts are not adjusted around the r2 acceptance questions. Selection-run or final data are not used.
- Passing the fictitious acceptance only shows that the label contract works on controlled inputs; it does not represent benchmark accuracy, correct entailment judgement or gains.

## 2026-10-26 pre-execution definition alignment

The new prompts and the 18+2 acceptance questions are unchanged; only the offline mode decision is aligned so that both datasets share one mode, and both NQ frozen EM and zero semantic harmful repairs are required. `gpu_release_v1` from `9d91a40` was not run and is kept as a historical sealed package; the run package with the updated definitions is regenerated before execution. For the Diagnosis new-interface merge task see [T1–T9 current tasks](CONTROLLER_TASKS.md). The development Full cost proxy does not enter the final confirmatory AC5 measured conclusion.
