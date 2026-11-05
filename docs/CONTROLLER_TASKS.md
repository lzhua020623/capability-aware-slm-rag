# Controller current tasks · T1–T9

2026-10-27. The user designated the October 24 Diagnosis `138d52a` as the main line going forward; see [new baseline registration](CONTROLLER_1024_BASELINE.md). The old T1–T5 remain historically frozen; the features, provenance and execution protocol of the new P60 are not yet ready, and switching the main line is not treated as effect acceptance. The original stage numbering is kept.

Subsequent development has moved to `controller_2`, compatibly merging retrieval enhancement `37e469e` and reranking `5583285`; see [merge handover](CONTROLLER2_INTEGRATION.md). NQ uniformly adopts the split of the team's `feature/baseline-evaluation@387ddaf`: Train 51,866, Validation 11,117, Test 11,114. Old configs and sealed packages are not overwritten.

This round's [T2 data alignment and boundaries](TEAM_DATA_BOUNDARIES.md) has integrated the original split, checking 74,097 IDs, the initial-label count metadata and the 15 local C1 shards; the current conservative allowed pool is Train 46,421 and Validation 9,440, with Test still the full 11,114. The known downstream test exposure check has been expanded to 603 items. Role and provenance checks have been added to the training/evaluation entry points; the new baseline per-question outputs, FEVER unification and the original used IDs of already-trained models are still to be delivered.

| Stage | Content | Status |
|---|---|---|
| T1 | Protocol, comparison groups, budgets, metrics and stopping rules | Frozen; amendments get separate versions |
| T2 | Data splits, common C1, retrieval resources, exposure isolation | NQ original split/ID/structural check and usage guard completed; known test exposure 603, conservatively usable Train 46,421, Validation 9,440; the old held-out pool is not unsealed; new baseline per-question outputs, model used IDs and FEVER unification still to be added |
| T3 | Integrated executor, budgets and trajectories, CUDA engineering acceptance | Retrieval enhancement 37e469e and the latest reranking 1d51856 compatibly integrated; the new selector/deep candidate pool keep experimental usage boundaries, default config unchanged; original online features/provenance and real new-main-line acceptance still to be added |
| T4 | 200-item development baseline for each of the two datasets | 2,800 items accepted; semantic, flag and trajectory human review not closed |
| T5 | Historical r2/routing selection run; new E distillation extension | Historical r2 stopped, E unavailable, 0.2 candidate rejected; the new T5-D independently prepares the protocol, 500-item teacher pilot production and development/selection-run IDs, without changing the production default |
| T6 | Controlled trajectory-driven self-improvement | Optional extension; resources evaluated after the main line |
| T7 | Final system, environment and evaluation package sealing | Current main line changed to the 10/24 Diagnosis; pending feature/provenance check, new protocol, budget and version sealing; the old routing is not carried over as default |
| T8 | One final paired evaluation on the team's unified test | Not started; NQ registers a new protocol with 11,114 items, FEVER pending unification; the old 500 items each are a historical protocol and are not silently rewritten |
| T9 | AC6, trajectory review, demo, report, team handover | Awaiting final closure |

## T5.0 · Layered merge of the Diagnosis update (authorized by the user)

The source is `138d52aa22e0eee78637e01f981a030d7ad388df` of `feature/diagnosis-probe-zheng`.

| Subtask | Requirement | Status |
|---|---|---|
| T5.0.1 | Review changes, model/feature provenance and the default path | Reviewed; original Diagnosis weights unchanged; the new P60 lacks training/tuning IDs and a real feature provider |
| T5.0.2 | Keep the old reranking contract and lazy imports, compatible with the new public interface and multi-failure representation | Integrated in this compatibility commit; does not replace the default experiment path |
| T5.0.3 | Add executor return consistency checks | Implemented with regression tests: action match, changed-answer flag consistent with reality, non-success results keep the original state |
| T5.0.4 | Merge into the committed r2, then run full regression and routing replay | Receipt in [compatibility integration](DIAGNOSIS_138D52A_INTEGRATION.md); GPU package re-sealed, original package kept |
| T5.0.5 | P60 current main-line integration | Selected by the user as the baseline going forward; current registration, real enhancement adaptation and strict checks implemented. The original 78-dim features and the training/tuning IDs of the three models are not delivered, so the production check stops; once supplied, register a new protocol and then accept |

The old T4/T5 continue to use their frozen `IntegratedController`/Probe path, and the single 0.2 candidate result is sealed. From 2026-10-27, the subsequent main line explicitly switches to P64/P60 of `src.pipeline.current`, Probe OFF; old experiments are not silently reinterpreted. While the original features and provenance are incomplete, there is no fallback to the old routing. Neither the multi-failure representation nor the new baseline selection equals a verified effect.

## T5.1–T5.9 · r2 and routing main line

| Subtask | Content | Current evidence and start conditions |
|---|---|---|
| T5.1 | Independent re-check and sealing of 18 short + 2 long questions | Completed; [sealing record](../artifacts/controller/e_v2a_interface_r2/fixture_review/README.md) |
| T5.2 | Register labels, acceptance branches, global mode and harmful-repair cap | Registered; aligned before GPU as a mode shared by both datasets, zero harmful repairs on frozen metrics, and NQ additionally requires zero semantic harmful repairs with no pending items |
| T5.3 | Implementation, tests, shared model decoding and sealed run package | Implementation, global mode and compatibility interface committed; the current v2 package references `e3ddead`, the original unrun package is kept |
| T5.4 | Kaggle 20-item real acceptance | Completed and audited: Version 1 / 362285592; 10 passed, 1 structural failure, 9 semantic failures, 0 execution failures, both long questions executed; not semantic_only |
| T5.5 | Development run | Executed the failed stop branch: 0 runs; the 400/800 development runs are not started, and no r3 is added |
| T5.6 | Independent ZIP audit and global E mode freeze | Completed: [raw ZIP and audit](../artifacts/controller/e_v2a_interface_r2/kaggle_v1/README.md), 26 checks consistent; all E groups of both datasets share unavailable, the real development compliance rate is not measured, FR12 not accepted |
| T5.7 | Human review (parallel) | [17-question confirmation snapshot](../artifacts/controller/t5_human_review_v1/review/review17.html): the user adopted all assisted opinions, 11 fully finalized, 6 not fully finalized; this is an assisted review with visible references. The remaining T4 changed answers and the historical 5 semantic harmful-repair IDs still await closure; 73 originally-correct kept questions are for regression only |
| T5.8 | Development replay, freeze at most one NQ routing candidate | Completed: [single 0.2 candidate](../artifacts/controller/t5_candidate_v1/README.md), NQ EM 78→81, 5 fixed/2 broken; semantic net value range −11 to +14. The user adopted the focus-queue opinions including pending ones; only frozen and sent to the selection run, not promoted |
| T5.9 | One selection-run confirmation | [First GPU raw output and independent audit](../artifacts/controller/t5_selection_v1/kaggle_v1/README.md) completed: 400 questions/2,000 records, zero call/budget failures; six active long paths passed. Original Adaptive: NQ 89→93, FEVER 163→164. The single 0.2 candidate relative to the original routing: NQ 93→90, 3 fixed/6 broken (frozen EM); promotion rejected per the registration and the original routing kept; no other thresholds tried, and the 15-pair semantic review is for explanation only. T8 not started |

The r2 results do not change frozen data, budgets or scoring. Comparison groups where E is unavailable are truthfully labeled as a keep policy; zero calls/zero harmful repairs cannot be used as evidence of repair capability or of AC6 semantic flag precision.

T5.7 current record: [user adopted assisted opinions](../artifacts/controller/t5_human_review_v1/README.md). The original [15-question unreviewed snapshot](../artifacts/controller/e_v2a_v1/t5_development_review/routing15_review.html) and [17-question assisted snapshot](../artifacts/controller/t5_assisted_review_v1/README.md) are kept; unconfirmed assistant judgments are still not filled in as human review, and user adoption does not eliminate pending values. The old T4, r1 and r2 failure results keep their original bytes.

## T5-D · E output distillation extension (2026-10-27)

The user authorized starting E distillation after the component review. The latest code has all been merged, but the P60 raw materials and the new weight acceptance are not yet complete, so the preparation without API/GPU is first completed independently; it is not claimed that all components can already run in production. See [E distillation preparation](E_DISTILLATION_V1.md).

| Subtask | Content | Current status |
|---|---|---|
| T5-D1 | Fix E input/output, the student and the original contract fingerprints | Preparation protocol and teacher execution budget sealed; training config and promotion values to be registered before training |
| T5-D2 | Freeze in-role samples | Train balanced pilot production 500; Validation uniform Development 500 / Selection 500; Test untouched |
| T5-D3 | Offline teacher tasks, reference isolation and local contract checks | Tools and API transport implemented; token-window v2 and visible progress registered separately, same ledger/budget/request body, requires explicit switching; full task replay, provenance/version/budget guards; passing the format does not automatically approve SFT |
| T5-D4 | Actual teacher registration and 500-item pilot production | 500/500 completed and the independent replay passed, zero failures/duplicates/unsettled; 49 replaced, 451 kept, longest target 126 tokens, conservative ledger $6.58152. The user batch-released 100 items in chat for candidate collection, with no per-item web annotation; SFT approvals are still 0. See [10/28 results](E_TEACHER_PILOT_RESULTS_20261028.md) |
| T5-D5 | Latest cumulative cap of 5,000 calls, filter qualified supervision data | The user changed to "go ahead with 5000", keeping the $40 cumulative cap. The 2,000 batch completed 54 items and stopped at a TimeoutError on the newly added 55th; the 554 complete results are reused, the 55th keeps its full reservation and is not resent, continuing from 56 with at most 4,445 calls. The 4500 Train tasks keep the original 1500 as prefix; no automatic backfill. Subsequent timeout 600 seconds; model/request body/zero retries unchanged. See [expansion and recovery](E_TEACHER_EXPANSION_20261028.md) |
| T5-D6 | QLoRA/SFT and real engineering acceptance | Not started; only train the E adapter, C1 fixed; first measure VRAM, long inputs and target token length |
| T5-D7 | Fixed-input comparison and one selection run | Not started; Development500 compares C1 / original E / distilled E, reporting semantic repairs/harmful repairs, EM/F1 and cost, and the student gain decides whether to consider expansion; Selection500 gives only the final candidate one confirmation |
| T5-D8 | Integration and version promotion | Executed once targets are met and P60/common baseline are ready; old experiments are not silently modified |

500 is the completed teacher pilot production; the latest user permission goes directly to the cumulative cap of 5,000 calls, including the no-response slot, i.e. at most 4,999 complete candidates, which cannot be written as the number of qualified SFT items or as the student being effective. The old 2,000 package and failure ledger are not overwritten; training/model selection cannot access Test, FEVER is not registered, and DPO/RSI are not started this round.

## Remaining acceptance for T7–T9

- T7: Fix code, mode, routing, team splits and exposure, environment, budgets and statistics. The original 500-item-each protocol stays historically frozen; extending to the team's NQ 11,114 items must be registered separately, and the already-exposed selection-run set is not reused as independent confirmation.
- T8: Paired evaluation on the team's unified NQ test of 11,114 items, with the FEVER scope unified separately; comparison groups, call quotas and statistical amendments are registered in T7 first. Per dataset, report frozen metrics, repairs, harmful repairs and intervals, and disclose known exposure. Final cost uses measurements; when the common C1 cost is incomplete, only the traceable scope is certified, and the development Full proxy cost is not used to declare full AC5 met.
- T9: Human review of at least 50 flagged items, AC6 observed precision target ≥90%, report coverage; if samples are insufficient, draw more in the pre-frozen order, without forcing flags. Additionally review at least 50 full trajectories, with the two items scored independently; complete the trajectory display, report, reproduction entry and responsibility handover.
- The report discloses requirements not yet passed, such as multi-label capability, FinDER and C1 provenance/cost; an E that is only verified or unavailable is not written as repair success.

## Resources and per-round self-check

The hard deadline and backup GPU are still to be provided. The most recent Kaggle check on October 26 showed `10:30 / 30 hrs`; planning assumes about 10.5 hours used, to be re-checked before launch. The final evaluation is roughly estimated at about 20 hours, requiring a quota reset or supplementary resources; it is not promised that the current quota covers the whole main line.

At the start of each round, check the branch, user files, frozen hashes, data roles, acceptance and budget; at the end, check the related tests and full regression, the independent result audit, the truthfulness of records and the commit signed only by the user. Source code is committed first, then the package is sealed; during runs the in-package source is authoritative. RSI, training/DPO, conditional automatic protection, independent verifiers and knowledge-base modification are each separate extensions, not mixed into the single main-line candidate.
