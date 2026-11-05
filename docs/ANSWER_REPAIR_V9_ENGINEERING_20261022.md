# v9 engineering trial run: evidence IDs and decomposed verification

This round is engineering validation on already exposed development data and cannot prove repair gains. The old v7/r3/r4 experiment bundles and results remain frozen; v9 is a separate entry point and has not yet been promoted to default policy.

`v9_dev_v1/v2/v3` are engineering snapshots and do not represent three new optimization algorithms; the research comparison has always been direct re-answering, evidence IDs, and evidence IDs + decomposed verification.

## Implementation and scope

Both FEVER and NQ use Qwen2.5-7B-Instruct at a fixed revision, NF4/FP16/double quant, seed 42, greedy. The three groups are direct re-answering, evidence IDs, and evidence IDs + decomposed verification. The latter first decomposes into at most three fact questions, then answers each by reading the full Top-5, and finally compares with C1; the first two stages do not see the original answer, and the fact Q&A calls do not see each other's outputs. They share the same model weights and cannot be called an independent verification model.

The direct group and the ID group make at most one call, the decomposition group at most five, each with at most 384 new tokens; the three groups are not compute-matched. Citations are reconstructed by the program from the source text by ID, and NQ changed answers must also be localized within the selected source text; the existence of a source cannot prove that the selected sentence is relevant or the answer correct. XGrammar, external web pages, NLI, training or automatic prompt evolution were not used.

Inputs are frozen as existing NQ75 and FEVER50 development samples; this time only five of each were taken by hash, excluding the retired NQ smoke8. The same ten questions run through three groups, expecting 30 predictions. The old challenge125 does not enter the new trial run and is not used for the selection run. Initial-answer human labels cover NQ75/75 and FEVER4/50; the five FEVER items this time happen to all lack human binary scores, so human semantic accuracy cannot be inferred from automatic reference answers.

## First version: engineering gate not passed

[Frozen inputs and collected files](../artifacts/answer_repair/v9_dev_v1/README.md). Kaggle Version 1 (`360999513`) failed after running 592.9 seconds: a CUDA OOM occurred in the final judgement of the second question of the FEVER decomposition group. The program did not switch models, switch quantization or truncate evidence.

26/30 predictions were saved: five groups complete with five each, and one in the last group. All five groups passed the local frozen-code replay, and the one saved record of the last group passed prefix replay; prefix replay cannot be called completion of the whole group. Of the 16 saved structured predictions, nine failed protocol validation and kept the original answer, including declaring keep but rewriting the original answer, repair not actually changing the answer, the answer not being in the selected source text, decision case mismatch, and the number of IDs exceeding the constraint. The ten direct-group outputs are counted separately.

This shows: source IDs can eliminate fabrication when the program reconstructs citations, but the model still selects wrong sentences, rewrites extracted answers or violates the contract. A low changed-answer rate caused by many fallbacks cannot be used to claim that verification is reliable. The costs of calls not yet written to results at the time of failure are missing; the sum of the costs of saved records is not the total cost of the whole failed job.

## Fixed version: compress only the evidence serialization

[Second frozen bundle](../artifacts/answer_repair/v9_dev_v2/README.md). The first version displayed repeated metadata such as `rank`, `passage_id` and start/end offsets for every sentence; the fixed version shows in the prompt only one passage mapping and `sentence ID → full source text`, while the full offset mapping is still saved in the trace. Characters, sentence IDs, near-duplicate conflicts, model, samples, rules, validation and budget are all kept. AST comparison confirms that only the `context` function changed in the repair module, plus an updated version identifier.

The context character count of the ten questions dropped to 58%–76% of the original, which is not equivalent to a proportional drop in tokens or GPU memory. The 13 core checks and related checks, 78 in total, passed, and the six-group frozen CPU pre-check passed. Kaggle Version 2 (`361004665`) ran 495.4 seconds in the same T4×2/internet-enabled environment and ended normally, 30/30 saved, all six groups passed local frozen replay, costs were bounded and did not exceed the pre-set budget, and no further OOM occurred.

But **19 of the 20 structured outputs were non-compliant, all fell back to C1, and zero actually changed the answer**; the remaining one was keep. This does not mean E has high accuracy or successfully avoided harmful repairs; rather, most outputs did not enter valid judgement. The ten direct re-answers are counted separately and cannot substitute for the success of structured E.

| Dataset/policy | Non-compliant/samples | Actual calls | Model generation seconds per question |
|---|---:|---:|---:|
| NQ direct re-answer | 0/5 | 5 | 1.26 |
| NQ evidence ID | 5/5 | 5 | 5.70 |
| NQ decomposed verification | 4/5 | 12 | 10.91 |
| FEVER direct re-answer | 0/5 | 5 | 4.10 |
| FEVER evidence ID | 5/5 | 5 | 9.24 |
| FEVER decomposed verification | 5/5 | 13 | 15.37 |

This run totals 45 calls and about 232.86 seconds of model generation time; the 495.4 seconds of job time also include dependencies, model loading, pre-checks and replay. Invalid outputs end the decomposition flow early, so this cost cannot represent the cost when the whole flow is valid.

Ten of them first reported Unknown evidence ID: the compressed format displayed both `sources.R1` and the citable `R1S1`, and the model chose the passage key. **This is an ambiguity introduced by this interface design and cannot be blamed solely on the model or the sample size.** Other problems include keep but rewriting the original answer, case, invalid JSON, citation count over the limit, and NQ answers still being rewritten sentences rather than fragments of the selected source text. Raw outputs are kept, and parsing is not relaxed to pursue a superficial validity rate.

## Single source-namespace candidate

A separate `v9_dev_v3` groups the evidence in the prompt by passage title and shows only the citable sentence IDs; full titles and source text are all kept. It adds explicit requirements for exact sentence IDs, at most five fact citations and lowercase decision; semantic rules, parser, fallback, model, the same batch of inputs and budget are unchanged. The 78 checks and the six-group frozen pre-check passed. The root directory is this candidate; reproducing Version 1/2 should use the source code inside each bundle.

Gate declared before submitting Version 3: 30 complete, six-group replay, valid budget, and at least 18/20 structured outputs compliant, before entering the next stage is considered. 90% is an engineering usability gate, not evidence of accuracy or significance. Results are frozen after this interface retest, and prompts are not further tuned around these ten questions; if still below the gate, record a negative result, shelve full development for now, and separately define a constrained decoding scheme.

Version 3 (`361008203`) ended normally in 523.2 seconds. All 30 are present, and the six-group local frozen replay and budget checks all passed; **source ID mismatch errors dropped from ten to zero, but only 11/20 structured outputs were compliant, not reaching the 18/20 gate**. The stop rule has been executed: prompts are not further tuned around these ten questions, and neither full development nor a new unseen evaluation is run.

| Dataset/policy | Compliant structured outputs | Actual changed answers | Calls | Model generation seconds per question |
|---|---:|---:|---:|---:|
| NQ evidence ID | 4/5 | 2 | 5 | 6.49 |
| NQ decomposed verification | 3/5 | 1 | 15 | 13.47 |
| FEVER evidence ID | 3/5 | 0 | 5 | 10.30 |
| FEVER decomposed verification | 1/5 | 0 | 13 | 15.76 |

The two direct re-answer groups made five calls each, about 1.30 seconds per question for NQ and about 4.09 seconds for FEVER; neither requires a structured contract, so their "no structured rejections" cannot be equated with the compliance rates in the table above. 48 calls in total, about 257.05 seconds of generation time. The remaining nine non-compliant: keep rewriting the original answer five, citation count one, invalid JSON one, NQ non-source answer one, repair not changing the answer one. Strict validation is kept, and raw outputs or human-review labels are not retroactively changed.

The three actual changed answers correspond to two questions, both NQ: only the ID group changed True Blood season seven from 16 to 10; the two groups each changed the actor in Harry and the Hendersons from Graham Greene to Kevin Peter Hall. The assisted opinion considers both questions' answers to be in the correct direction, but there is no new human confirmation in this round yet, and they are not counted as three independent human-reviewed repairs.

True Blood exposes a more important boundary: R1 explicitly says ten episodes; the model output 10 and cited R1 and R2, but the program found the literal answer at the **June 10** date in R2. R2 actually says 16 episodes and does not name True Blood, yet the model's reason claimed R2 likewise confirms 10 episodes. That the answer happens to be correct, the citation comes from the source, and literal localization succeeded does not mean the selected answer fragment corresponds to the correct factual relation. This round does not add a special-purpose rule for this question.

The new changed answers have been organized into the [V1–V3 assisted review web page](../artifacts/answer_repair/v9_dev_v3/kaggle_20261022_v3/changed_review.html) and `changed_assisted_review.jsonl` in the same directory. Original-answer errors carry over the previously confirmed NQ75 labels; only the new candidates and citation quality are left for human confirmation; this is separate from the closed F42 list.

## Follow-up plan for output constraints (not implemented in the first three frozen bundles)

Prioritize a separate constrained generation scheme, validating the interface with new synthetic engineering examples, without further tuning on these ten questions. Based on the [official XGrammar HF integration guide](https://xgrammar.mlc.ai/docs/latest/start/quick_start.html) and the [JSON Schema documentation](https://xgrammar.mlc.ai/docs/latest/defining_structures/json_generation.html), the plan is to constrain fields, decision/FEVER labels and the currently valid sentence IDs at generation time, and to have the contract constrain the answer for keep to be the original answer. This is a proposed design of this project; none of the current three GPU bundles integrate XGrammar.

The Qwen NF4/current Transformers/Python combination needs separate validation, recording compilation and generation costs; limiting output length may still cause incomplete results. The program continues to do exact source, count/deduplication and extraction checks. A valid source ID and valid JSON cannot prove that the subject, relation, time or answer is correct; candidates should be bound to specific supporting fragments, and numeric representation normalization should be designed separately from semantic judgement and validated independently.

Later that day a separate [contract_v1 experiment](../artifacts/answer_repair/v9_contract_v1/README.md) was added, using 12 brand-new fictional questions, two structured policies and two decodings, prompt_only/XGrammar, 48 predictions, without reusing the ten questions above. Both decoding groups received the same schema text, with fixed Qwen revision/NF4/budget. Kaggle361060970 completed 48 predictions and the cloud replay; the 40 local source hashes, 48 replays and the summary recomputation all passed; 96 calls in practice. The constrained group was 22/24 strictly compliant, reaching the independently pre-declared 22/24 gate, with Unknown ID and keep rewriting at 0; the prompt group was 20/24. The added compliance corresponds to two policy results for the same S02, and the remaining constrained rejections correspond to the 9/nine source-fragment restriction of the same S03; they cannot be treated as changes across multiple independent questions.

This task errored at the final ZIP export: frozen source mtime=0, and ZIP by default rejects timestamps before 1980. The complete raw results were recovered via Kaggle's Download output and the full log saved; inference and replay were not lost. The new exportfix notebook only fixes the export timestamp handling; the original bundle and original notebook are not overwritten, and the model is not rerun. The platform failure and inference completion are recorded separately.

Passing the new engineering gate only permits separately freezing a real semantic development protocol; it does not indicate an accuracy improvement or default-policy promotion. The old ten-question stop rule remains in effect; prompts will not be further tuned or extraction conditions relaxed around these 12 questions. Label quality review, necessary human review and scoring trade-offs remain the next dependencies.

## Subsequent order

First settle the last interface retest, and decide per the pre-declared gate whether to enter formal development. Passing execution and replay does not mean the output contract or semantic gate passes. Before the full development run, fix the semantic scoring and the wrong-rejection/harmful-repair/cost trade-offs, and add the necessary human scores for FEVER and candidate answers; reducing format rejections to zero cannot be treated as repair gains.

Then run the fixed development comparison and review changed answers; after freezing, prepare new unseen data and confirm training exclusion and near-duplicate isolation with the Diagnosis owner. The existing 125 challenge has already been used and cannot be repeatedly used for prompt tuning. Historical H1–H45 can be reviewed in parallel; F1–F42 are processed, and the user is not asked for conclusions again.
