# E v2a and the single NQ routing candidate · 2026-10-25

This amendment was registered after observing T4 and before running v2a and searching routing. T1/T3, the original T4 predictions and gold keep their original bytes. The E interface correction and routing gains are evaluated separately, and the amendment is not passed off as the original pre-registration. The machine protocol is in [e_v2a_amendment_v1.json](../configs/controller/e_v2a_amendment_v1.json).

Current: registration, the single-call interface, the shared model's XGrammar view, metering/fallback, GPU prerequisite gates and sealing tools are implemented. [The actual results of interface correction r1](../artifacts/controller/e_v2a_interface_r1/README.md) are sealed: the fictitious NQ keep and repair cases passed, while the verification category of the missing-evidence case did not match expectations; it was structurally valid and safely kept, not a harmful repair on that question. It was stopped per the rules; real development predictions are 0, FEVER and long inputs are untested, and the requirement to get E working end to end has not yet been accepted; the original T4 is not overwritten. [The 15-question review page](../artifacts/controller/e_v2a_v1/t5_development_review/routing15_review.html) contains the original Top-5, the actual evidence for each action and offline references; human review is still 0/15, and the routing candidate is not frozen. The ID verification of the five semantic harmful repairs is still pending user confirmation.

2026-10-26: [r2 unified label definition](CONTROLLER_E_V2A_R2.md) registered separately. r2 is a semantic-definition amendment; r1 results are kept as is. 18+2 new fictitious acceptance questions were sealed after three rounds of independent blind review before the prompts were changed; the acceptance run executes all checks and classifies failures separately into execution, structural and semantic categories, and the development stage proceeds along three branches; the main line does not schedule r3.

Development replay of three registered thresholds: 0.2 gives 5 fixed/2 broken, net EM +3, 107 additional retrievals; 0.3 gives 2/1, net +1, 64 additional; 0.4 gives 0/0, 26 additional. For design only; no candidate has yet been frozen or promoted; semantic gains are pending human verification; see [all attempts](../artifacts/controller/e_v2a_v1/t5_development_review/routing_search_development.json) for details. The module parameters, inputs and fixed-action outputs of the 800 historical Adaptive/No Probe items are all consistent, and the evidence is kept separately.

2026-10-25: after Chrome was restored, Kaggle rejected the embedded Notebook (source over 1 MB) before allocating a GPU. The run entry point is now the [transport_r2 short Notebook](../artifacts/controller/e_v2a_v1/transport_r2/controller_e_v2a_development_v1_dataset.ipynb), about 4 KB, which reads the original compressed bytes with a `.bin` suffix from a private Dataset to avoid automatic decompression. The original tar SHA256 is still `728c8ba512115825c3a4c362f5e4a51a3c860d74ad4d6e977dcaf93ca12adcf3`; inference source, prompts, budget and gates are unchanged; this is a non-semantic retry. The original embedded and intermediate transport files are kept only for audit. After adding two transport checks, the full 811-item regression passes.

## Start-of-work and end-of-work self-checks

Each logical work round first checks the branch, the user's existing files, frozen fingerprints, data roles, acceptance conditions and GPU quota; at end of work it checks tests, diff, frozen fingerprints and the truthfulness of records, and commits with the single configured user identity. Pre-existing untracked annotation files are not included in this round's commits. CPU checks cannot replace real CUDA receipts, and assisted opinions cannot replace human review.

## Engineering boundaries of E v2a

Same Qwen2.5-7B/NF4, single call, 384 output tokens; NQ keeps v7's 20-word candidate cap. Reuses v9's lossless sentence numbering, with quotes restored only by the program and no gold at runtime. Adds explicit original-answer verification and candidate checks, without introducing multi-round fact decomposition, qualifier structures, independent models or training.

`supported` is always kept; `unsupported` only accepts changed answers that pass the source and answer-span checks, otherwise the answer is kept and marked unresolved. `insufficient` is kept, with evidence conflicts recorded in the reason. Failures, truncation and unparseable original-answer verification are marked `incomplete`, without fabricating a judgement. When the candidate is unqualified, the independently verifiable original-answer verification fields are still kept. An equivalent expansion does not count as a semantic repair; a wrong answer that becomes right after adding a necessary qualifier can count as a repair; EM and semantics are reported separately.

The engineering prerequisites use three brand-new fictitious questions per dataset, with three expectation classes fixed in advance (supported-keep, error-repair, missing-evidence-unresolved); all passed. Each dataset separately runs at least a 14k-token long input with XGrammar 0.2.8 and prefill256 both enabled; silently turning off either feature is not allowed. A prerequisite failure stops immediately, without entering the real development set.

The implementation references the [official XGrammar HF interface](https://xgrammar.mlc.ai/docs/latest/start/quick_start.html); its structural-constraint capability does not equal correct semantic verification, nor can it replace actual CUDA evidence for this project's NF4/prefill256 combination.

Real development uses 200 items per dataset, with a full contract compliance rate of at least 90% for each; failures and fallbacks are not removed from the denominator. When the proportion of the union of insufficient or unresolved strictly exceeds 90%, enabling is paused and re-examined; the proportion cannot be pushed down by forcing changed answers. This is a debugging line, not ground truth for judging reasonable refusals as errors.

## Selecting E's run mode on the development set

First save the repair-allowed diagnostic output once, then derive the unified run mode offline, without regenerating and picking answers. If either dataset fails compliance, engineering does not pass and the mode is fixed as unavailable: keep the original answer and mark verification as incomplete, without choosing post hoc between v1 and other fallbacks.

After compliance, compute the net change per dataset on the E subset of the frozen T4 Adaptive. NQ has 16 items, FEVER zero; this set does not change because of v2a results. A non-negative net value allows changed answers; a negative or non-estimable value means verify_only. FEVER having no E-routed samples cannot be assumed safe; the default is verify-only. All system groups containing E use the same frozen per-dataset mode; Full/Always must explicitly disclose that E replacement is not actually executed when in verify-only.

The denominators of `p*r-(1-p)*h` are the original-wrong/original-right of the same reference set; NQ uses frozen EM and does not pass it off as utilisation semantic labels. The historical 9/32≈0.28 comes from the target-type/correct ratio of the already-exposed PRIMARY and can only serve as a historical risk note, not be applied directly to all datasets. Passing the mode gate does not mean a statistically proven positive gain.

## Development runs and replay

Always Repair v2a: 400 runs across both datasets in total; in addition, 400 runs on the actual E-entering states of the original Full. Full's first two stages can be reused, but single-module results on the original C1 cannot be transplanted onto the changed states. Save entering-state, version and budget fingerprints. Adaptive/No Probe reuse the fixed-action outputs on the same inputs; the original Probe calls keep their cost, and replay is not a measured deployment latency.

Development reads only development gold; selection run and final are not scored in advance. The five real E harmful repairs and the 73 original-right kept questions pointed out by the user are used only for debugging; while per-ID review evidence is incomplete, no formal settlement is claimed. The 15 key NQ routing cases must be confirmed by the user before the candidate can be frozen.

## Single routing search

Applies only to NQ with original action=no_action. The single variable is the lower bound on first-level failure probability, with the grid fixed at `[0.2, 0.3, 0.4]`; reaching the lower bound triggers retrieval enhancement. An existing explicit supported Probe always vetoes extra retrieval; without a Probe, the existing Diagnosis probability is used directly, no Probe is added and missing judgements are not forged. The model's original labels are kept, and policy overrides are recorded separately.

T4 has 153 items whose action can vary; metrics still cover all of NQ200. Record three attempts and choose in the order of EM net gain, fewer harmful repairs, fewer extra retrievals, higher threshold. If the best net gain is not positive, the original routing is kept. All threshold choices use only development; runtime rules must not access gold, human-review correctness or special-case by ID.

The candidate is replayed once more on the new baseline with v2a's fixed mode. After human confirmation of the 15 key cases, only one rule is frozen to enter selection. The selection run runs only the four fixed-action/Adaptive groups, with C1 cached; Full, No Probe and E v1 are not run. Existing Probe traces are collected only once, and the two routing sets are compared under matching conditions. NQ frozen EM net gain must be positive; it is vetoed if semantic net regression or unresolved items could turn the semantic net value negative. FEVER's routing is guaranteed unchanged by the dataset scope, and output consistency is accepted.

## Final and delivery

After freezing, 500 items per dataset, the six main groups are run only once, following T3's four tests/Holm and the complete cost definition. Estimated new development 2–3 hours, selection run about 4.4 hours, final about 20 hours, with additional margin for loading and interruptions; actual quota is as per the account. The two semantic extensions, training, cleaning and RSI come after the main line.

Each round's delivery states the completed code/CPU checks, the actual GPU status and the fields pending human review; packaging, submitting a Notebook or generating valid JSON is not treated as an accuracy improvement. AC6 and complete trace review are accepted independently per the original protocol.
