# E experiment results and handoff (2026-10-13)

The E engineering pipeline, real T4 inference, and the frozen Controller→E integration all run end to end; **algorithmic effectiveness has not yet passed acceptance**. Current results do not support the claim that "citation checking plus same-model regeneration can reliably improve answers". v4 should be kept as a reproducible baseline; the next step is to solve query matching and semantic support, and then evaluate utilisation samples and misrouting risk.

## Experiment scope and frozen protocol

- C1 source: `6f08f2fe2dc2dcf39b75177763207682b2890a31`.
- Diagnosis / Probe / Safe Binary / Controller: `7c7863f98f87e892119481a79e23ca6dcebdced0`; no retraining or threshold changes in this run.
- Model: Qwen2.5-7B-Instruct, revision `a09a35458c702b33eeacc393d103063234e8bc28`; NF4, double quant, FP16 compute, greedy, seed 42.
- Hardware: Google Colab Tesla T4. 196 NF4 linear layers were actually detected, with parameters on CUDA; allocated GPU memory after model loading was about 5.56 GB, which is not the full generation peak. A T4 is sufficient for this quantized inference run.
- Fixed development samples: 50 each from FEVER / NQ, selected from the training intersection of Diagnosis v1/v2 by ID hash with seed 42; all validation/test IDs of both versions are excluded. Sample selection does not use answer correctness, gold, or diagnosis labels. The 10 smoke samples are a subset of the respective 50 and cannot be combined into more independent samples.
- Both E and Reanswer use the original C1 Top-5, with at most one extra generation per sample, capped at 384 new tokens. At runtime, gold, C3, 70B, and human diagnosis are not read; gold is used only for offline scoring after generation.
- This is an exploratory development evaluation on training samples and cannot be used to report held-out performance. The historical C1 did not record the model revision, so we cannot claim its environment is fully identical.

Environment record: Python 3.13.15, torch 2.11.0+cu130, CUDA 13.0, transformers 5.18.0, accelerate 1.15.0, bitsandbytes 0.50.2. The Controller run used numpy 2.1.3 / scikit-learn 1.6.1.

## Always-E vs. Reanswer comparison

The development strategy `grounded_reanswer` hides the initial answer and has the model produce answer/citations from the query and evidence; the program then compares the candidate with the initial answer, derives a verdict, and checks the citations. `reanswer` re-answers with the original C1 instruction and produces no structured citations. Neither is an independent verifier.

| Dataset | C1 correct | Grounded Reanswer | Fixed / broken | Reanswer correct |
|---|---:|---:|---:|---:|
| FEVER accuracy | 45/50 | 46/50 | 1 / 0 | 45/50 |
| NQ EM | 20/50 | 20/50 | 1 / 1 | 20/50 |

NQ token-F1: C1 `0.548912` → Grounded Reanswer `0.544144`; Reanswer is `0.548912`. The exact McNemar p-values for paired correctness changes are all 1.0; this batch provides no evidence of a significant gain.

| Run | Answer changes | Invalid outputs | Kept / insufficient / repaired | Extra calls | Total generation latency |
|---|---:|---:|---|---:|---:|
| FEVER Grounded | 1 | 3 | 15 / 31 / 1 | 50 | 185.446 s |
| NQ Grounded | 8 | 14 | 18 / 10 / 8 | 50 | 221.673 s |
| FEVER Reanswer | 0 | 5 | 45 valid re-answers | 50 | 120.010 s |
| NQ Reanswer | 1 | 0 | 50 valid re-answers | 50 | 64.137 s |

Both `insufficient_evidence` and `invalid_output` fall back to the initial answer; these records cannot be counted as verified correct. Latency is the extra model generation time and excludes model loading, the original C1, full retrieval, or whole-system cost. Output tokens are re-tokenized estimates.

## Key cases: answers can still be wrong despite citations

The following is Codex's qualitative review of the fixed development outputs, **not a human gold standard, and not an independent runtime semantic verification**. The original scores and raw outputs are unchanged.

| ID | Change and observation | Implication for E |
|---|---|---|
| FEVER `219125` | Valencia changed from REFUTES to the gold SUPPORTS; the citation comes from Venezuela's `Valencia_Municipality,_Carabobo` and cannot support the Spanish Valencia the query refers to. The existing diagnosis is insufficiency. | Metric +1, but it cannot be regarded as a trustworthy evidence-supported repair. The entity and the full claim need verification. |
| NQ `-5748368670037082784` | Asks the meaning of the name Thomas; the original answer `twin` is correct; it was changed to an introduction about medieval popularity and biblical figures, and EM/F1 dropped to 0. The citation is valid. | A clear harmful repair where query and answer do not match; text appearing in the evidence does not mean the question was answered. |
| NQ `-1814654291537702625` | The Christianity case expanded from `love` to the full Great Commandment sentence, EM 0→1; the existing diagnosis is correct. | Possibly alignment with the reference answer text; it cannot be directly concluded that an original semantic error was fixed. |
| NQ `-2282501421613384225` | Asks about tibia/fibula; the candidate still describes the location of only one of the bones; EM did not improve. The citation is valid. | Answer-span checking cannot verify the relation and coverage the question requires. |
| NQ `4305916235259203400` | Asks the start/end years of the Soviet Union; original answer `1922-1991`; frozen gold is `in Eurasia`. The candidate used a non-contiguous spliced citation and was rejected. | Flag the reference answer issue separately; do not modify gold or the scorer to obtain gains. |

Among these 100 development samples, only 5 have an existing diagnosis annotation of utilisation. Always-E did not repair them: Sugarland Express, thumbs twiddling, and the Annie singer produced insufficient; Loihi produced invalid output; tibia/fibula was still wrong after rewriting. Some original Top-5 already contain the direct answer, e.g. the director of Sugarland Express and the singer of Annie, showing that candidate generation itself also over-abstains. The sample is too small to extrapolate an overall repair rate.

FEVER invalid outputs: 2 citation mismatches, 1 missing citation. NQ invalid outputs: 7 citation mismatches, 7 answers not in the citation. The latter include real wording differences, splicing, and strict-matching issues with hyphen/punctuation spacing; the normalization rules can be fixed separately later, but a new version must be released and this run's results must not be retroactively changed. A valid citation only means the source matches; `independent_support_status` is `not_checked` for every record in this run.

## Real frozen Controller→E

Using the same 100 development samples, the frozen classifier, selective Probe, Safe Binary, and Controller were actually executed, and then the E adapter consumed the final `ControllerDecision`. Probe and E share one quantized model. The CPU fixture contract validation and this real GPU run are saved separately.

| Dataset | no_action | retrieval_enhancement | evidence_refinement | generation_repair | Probe calls | E calls | C1 → final correct |
|---|---:|---:|---:|---:|---:|---:|---:|
| FEVER | 31 | 19 | 0 | 0 | 14 | 0 | 45 → 45 |
| NQ | 40 | 7 | 2 | 1 | 13 | 1 | 20 → 20 |

The only E route was the Soviet Union case, whose output was invalid and fell back; all 5 annotated utilisation samples were routed to no_action by the final Controller, with no Probe executed. This reflects both an E capability problem and a routing problem in which target samples never reach E; neither should be masked by modifying the frozen upstream thresholds.

In total 27 extra Probe calls + 1 E call were made. Probe generation took about `177.722 s` and E about `7.569 s`; Probe input tokens 33,311, estimated output 1,915; E input 1,100, estimated output 98. **The other two compensation actions were only recorded, not executed, so this is not a complete adaptive framework result.** The routing distribution of this training development cohort also cannot replace the upstream held-out Diagnosis evaluation.

## Validation and reproducible artifacts

Using the code inside the frozen archive, all 240 v4 outputs (40 smoke + 200 development) were replayed one by one on CPU, checking input IDs/order, code and data fingerprints, parsing/citations, scoring, cost, and summaries; the 100 Controller samples additionally replayed the actual classifier, Probe parser, Safe Binary, decision, and E fallback. The floating-point tolerance for summaries is `1e-12`. 54 tests in the main code pass. The above validation verifies that the implementation matches the records; it does not prove semantic support.

Full results package: [E_answer_repair_20261013_results.zip](../../../artifacts/answer_repair/session_20261013/E_answer_repair_20261013_results.zip). Raw data, code archives, logs, environment, and checksums have been downloaded locally. `artifacts/` is ignored by Git; code and documentation changes have not yet been committed or pushed. Combined package SHA-256 (computed locally): `cf8dfd42fc949e39bb4dfa66c1307149178565dc5c2ad9b1b126b560b8368e5a`.

| Package | SHA-256 |
|---|---|
| v4 GPU code and inputs | `2a40d4fbaff591b02d116243f92404731cb71ccc5ce8a77092f4fda70dd3156e` |
| Frozen Controller + v4 E | `60c4241e8a800ecbd8d780a913cbaf137927bbef7f954cadd16bd58ea7f7506c` |
| v4 smoke export | `e961bec18592bf2371a716cf87657c5f4eb83c569b595a3025013d9c52ddb27b` |
| v4 development export | `7a4def101c1ce5d524b941b51ba00e94a05b0b5d4ea471ed601a9fa59627dc72` |
| Controller→E export | `b28533f874786836ee9c92fd4f1a084dda64723068fd030daf38be18fbd36d55` |

Readable local records:

- [Always-E / Reanswer outputs and validation](../../../artifacts/answer_repair/v4/colab_20261013_development)
- [Real Controller→E outputs and validation](../../../artifacts/answer_repair/controller_20261013)
- [Colab / Kaggle execution method](../../ANSWER_REPAIR_COLAB.md) and [v4 notebook](../../../notebooks/history/answer_repair/answer_repair_colab_v4.ipynb)
- [E protocol](../../ANSWER_REPAIR.md), [citation and output checks](../../../src/compensation/answer_repair.py), [model generation and prompt](../../../src/compensation/qwen_repair.py), [Controller adapter](../../../src/compensation/controller_adapter.py)

## Next steps and acceptance criteria

1. Freeze this v4 and its negative results. Separate "citation source is valid", "the query is answered", and "the evidence supports the conclusion", and add semantic verification that can reject candidates like Thomas / Valencia. If the same 7B is still reused, it must be validated through a harmful-repair-rate ablation; it cannot be called independent verification.
2. Run operation-level/oracle-route evaluation on a utilisation-stratified cohort within training that has not yet been used for prompt tuning, while keeping originally correct and misrouted samples. Measure E's repair ability, harmful repair rate, abstain/invalid, and cost separately; oracle diagnosis is used only as an offline ablation and is not fed into the deployment pipeline.
3. Report E capability separately from frozen routing coverage. None of the current 5 target cases reached E; these cases need to be handed over to the diagnosis owner, whose independent experiments decide whether to release a new upstream version.
4. After the three compensation modules are merged, measure the failure-type × action matrix and the Vanilla / Always / Adaptive comparison, and record the full cost of retrieval, feature extraction, Probe, and compensation.
5. Reconfirm that the final evaluation data were not used for strategy selection in any module. The new `reranking` branch compared keep3/keep5 on NQ frozen500 and then chose keep5 as the default; these 500 include 65 validation and 68 test IDs of Diagnosis v2. That reranking result should be treated as exploratory development evidence, and unseen-test gains of a new adaptive system cannot be claimed on the same batch. The existing frozen test metrics of Diagnosis itself are not recomputed because of this.

Only after E demonstrates trustworthy evidence-supported repair, controlled harmful repair, and reasonable cost in an independent evaluation can the algorithm be declared complete. If there is still no net gain, the project can honestly report the applicability boundaries of the compensation operations and the routing mismatch; there is no need to manufacture positive results to match the proposal.
