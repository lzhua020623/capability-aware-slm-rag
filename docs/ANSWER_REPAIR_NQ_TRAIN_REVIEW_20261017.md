# NQ Train development sample assisted review: 2026-10-17

As of the sixth batch of opinions on 2026-10-19, **all 100 records of the fixed first batch have one round of human opinion; 75 finalized (11 E utilisation targets, 64 correct controls), 25 pending, 0 not yet viewed**. Completing the first viewing round does not mean all main labels are finalized, let alone that all 74,097 records are labelled. The 25 include the original 17 deferred, the 5 completeness/expression criteria disagreements of the fourth batch and the condition or supplement requirements of N58/N61/N83; the user's original words and the assistant's re-check are kept separately and not automatically treated as factual errors. The 64 controls include the same-entity source wording check of N76, marked separately. The original data follows `origin/main@058f10baa47083220f95edb39db8f106aa7e85d6`; no new main commits after this commit were found at the earlier check, and the remote was not re-queried this round.

For the current status see the [cumulative 100-record review record](../data/annotations/human/answer_repair_nq_train_dev_v1_through_batch06_20261019.jsonl) and the [cumulative counts and pending groups](../data/annotations/human/answer_repair_nq_train_dev_v1_through_batch06_20261019.summary.json). The records of the first five batches and the [fifth-batch cumulative historical counts](../data/annotations/human/answer_repair_nq_train_dev_v1_through_batch05_20261019.summary.json) are kept unchanged. These are conversational opinions given after viewing the assisted analysis and evidence, not independent blind labels; no new E output has been confirmed correct.

At integration on 2026-10-17, the 11 delivery/pipeline files newly added on main were checked against the local versions and all are byte-identical; the 15 original shards downloaded from Google all match the SHA-256 in the main manifest. The 74,097 source records and their gold were not modified. E's conversion files, internal partition and review suggestions are all derived products; if they conflict with later formal protocols on main or a shared team partition, a new derived version should be rebuilt from upstream.

## Sixth batch of human opinions and first-round closure: 2026-10-19

The user explicitly confirmed that N86/N87/N88/N90/N92/N93/N95/N97/N100 are all correct, and all 9 were added to the correct controls. The original words and scope are in the [sixth-batch confirmation](../data/annotations/human/answer_repair_nq_train_dev_v1_batch06_20261019.confirmations.json), and per-record entries are in the [sixth-batch human record](../data/annotations/human/answer_repair_nq_train_dev_v1_batch06_20261019.jsonl). All source records, passage citations and character position checks pass; the advantage summary of N87 and the protocol/specification wording of N88 are accepted only for these cases, and no global short-answer rule is inferred.

| First-round outcome | Count | Use |
|---|---:|---|
| Internal E utilisation targets | 11 | Check during development whether the original error is repaired |
| Correct controls | 64 | Check during development whether an already correct answer is harmfully repaired |
| Pending | 25 | Isolated, not used in current repair gain statistics |
| Not yet viewed | 0 | All 100 records have received human opinion |

The currently usable development inputs are the [75-record manifest](../configs/splits/answer_repair_nq_train_confirmed_dev_v6.json), the [frozen C1 subset](../artifacts/answer_repair/nq_train_dev_v1/confirmed/through_batch06_c1.jsonl) and the [pure runtime subset](../artifacts/answer_repair/nq_train_dev_v1/confirmed/through_batch06_runtime.jsonl). All 75 come from development, the calibration/reserved intersection is 0, and the runner's frozen scoring precheck passes. The 25 old pending records stay consistent record by record and in original byte content, and are not automatically released because the first round ended. Gold, original C1, the frozen scorer, upstream Diagnosis labels, v7 results and the E algorithm were not changed, and no new GPU experiment was run.

The pending list is divided into three groups by source:

- Original disputes unresolved, 17 records: N3/N7/N19/N28/N44/N56/N57/N59/N63/N67/N78/N79/N82/N89/N91/N98/N99.
- Completeness/expression criteria, 5 records: N13/N18/N22/N41/N43.
- Purchase conditions, location scope or time qualification, 3 records: N58/N61/N83.

The next round can use the 11 targets and 64 correct controls to advance controlled development of original answer checking, minimal repair and candidate checking; the 25 remain isolated, and forcing a classification first is not required. Changed to correct, harmful repair, no change, parse failure and call cost must each be recorded separately; this subset was selected through human review and is not an unseen test set, so it cannot be directly extrapolated to overall gain. This round only completed review consolidation and input preparation, and does not mean v8 has been implemented or is effective.

The latest web import file is the [sixth-batch cumulative status](../artifacts/answer_repair/nq_train_dev_v1/review/NQ_train_pending_92.chat_confirmations_batch06_20261019.json); the original 92-record package shows 67 confirmed, 25 pending, 0 unreviewed; the other 8 earliest confirmed E targets were never on that page. Back up other browser drafts before importing; the file contains the opinions of the previous batches, and the original data package hash is unchanged.

## Fifth batch of human opinions: 2026-10-19 (historical)

This batch has 22 records in total, 18 explicitly judged correct: N51/N53/N54/N55/N60/N64/N65/N66/N68/N69/N70/N71/N72/N73/N75/N77/N80/N85. Per-record citation positions, passages and source hashes all pass the check; recorded as of the frozen corpus time, the current of N51 and the new of N80 are not interpreted as today's latest version or person status.

| ID | User opinion and record |
|---|---|
| N58 | The user judged 18 wrong and proposed 21; the original words and the negative ruling are kept. R1 states both the general non-NFA purchase threshold of 18 and 21 for buying a handgun from federal firearms licensees; the question does not restrict firearm type/channel, and gold is 18 years. Recorded as a purchase-condition dispute, the full label is deferred, and 21 is not used directly to cover all scenarios |
| N61 | The user wants Toowoomba, Queensland, Australia written out. R1 explicitly gives the main campus location, but also supports the three campuses Toowoomba/Springfield/Ipswich listed by C1. Recorded as a requirement for the full location and main campus/campus scope, not classified as a factual error for now |
| N76 | The user said "Dutch St. Nicholas"; checked against R1's Dutch St Nicolas traditions and R2/R3's Dutch Sinterklaas as the same source wording, kept as a correct control. This record is marked as a same-entity interpretation and does not falsely claim the user directly wrote "correct"; the historical Greek bishop is not described as Dutch |
| N83 | The user requested a time qualification; R1's "as of July 2018, Instagram's own account" is recorded. The original entity is supported under the frozen corpus criterion; whether lacking a time statement makes the original answer wrong is not yet clear, so the time-completeness requirement is recorded first, the full label is deferred, and today's ranking is not queried or substituted |

This batch's opinions are fully saved in the [fifth-batch confirmation scope](../data/annotations/human/answer_repair_nq_train_dev_v1_batch05_20261019.confirmations.json) and the [per-record entries](../data/annotations/human/answer_repair_nq_train_dev_v1_batch05_20261019.jsonl). This batch yields 19 correct controls (18 explicit "correct" plus N76's equivalent source explanation), with no new E core targets; N58/N61/N83 are listed separately as pending. The 5 criteria disagreements of the fourth batch were not automatically resolved by continued review or the passage of time.

The cumulative development manifest is [v5: 66 records](../configs/splits/answer_repair_nq_train_confirmed_dev_v5.json), corresponding to the [C1 subset](../artifacts/answer_repair/nq_train_dev_v1/confirmed/through_batch05_c1.jsonl) and the [runtime subset](../artifacts/answer_repair/nq_train_dev_v1/confirmed/through_batch05_runtime.jsonl), containing 11 E targets and 55 correct controls. All come from development, the calibration/reserved intersection is 0, and the frozen scoring precheck passes; no gold, scoring rule or original C1 content was modified and no GPU was called. The sources keep both the user's opinions and the assistant's disagreements, and review counts cannot be treated as repair gains.

The web page can import the [fifth-batch cumulative status](../artifacts/answer_repair/nq_train_dev_v1/review/NQ_train_pending_92.chat_confirmations_batch05_20261019.json), which shows 58 confirmed, 25 pending and 9 blank on this page. It contains all opinions of the previous batches within the web page scope; back up other browser drafts before importing; the original 92-record package and hash are unchanged.

Only N86/N87/N88/N90/N92/N93/N95/N97/N100 remain unreviewed.

## Fourth batch of human opinions: 2026-10-19 (historical)

In this batch the user gave opinions on 38 records, of which 33 were explicitly judged correct: N1/N2/N4/N5/N9/N10/N11/N12/N14/N15/N16/N17/N20/N21/N23/N24/N25/N26/N27/N30/N32/N34/N35/N36/N37/N38/N39/N40/N42/N45/N46/N47/N50. Recorded as correct controls, without further presuming that the user confirmed a separate `initial_supported` score. All source records and citation positions pass the check.

The original words and assistant re-checks of the other 5 are saved in [fourth-batch original opinions and criteria to clarify](../data/annotations/human/answer_repair_nq_train_dev_v1_batch04_20261019.confirmations.json) and the [per-record review entries](../data/annotations/human/answer_repair_nq_train_dev_v1_batch04_20261019.jsonl):

| ID | User opinion | Re-check and handling |
|---|---|---|
| N13 | Did not answer "on the snow" | The question already has in the snow; C1 and gold both give lying on the back and the arm and leg movements. Adding the snow makes the statement more self-contained and complete; whether this constitutes a semantic error awaits criteria confirmation |
| N18 | Not comprehensive; should answer goal/purpose | C1 answers the core definition, and the question asks meaning; R1/R2's understand how the universe behaves can supplement the goal. Whether a definition question must include the goal awaits confirmation |
| N22 | Doesn't count as correct; answered an action, should answer a noun | The original words "doesn't count as correct" are kept in full, and `initial_correct=false` only represents the user's formal ruling. C1 is actually a noun phrase with a participle modifier, already listing Irish cream, whiskey, stout; the assistant keeps the opinion that it is semantically acceptable, and it is not included in E as a factual error |
| N41 | Did not say lowercase | C1 actually contains the lowercase glyph π. lowercase can be added, but the glyph itself already expresses the case; whether an explicit modifier is required awaits confirmation |
| N43 | Should write the city of Jodhpur, Rajasthan | R1 supports Jodhpur, Rajasthan; C1 already gives the correct city. The user's city + state expression preference is kept, and the location granularity standard awaits confirmation |

A request has been sent to the user to align the evaluation criteria for these 5. Until a reply to that question arrives, the per-item opinions and the assistant's disagreements are kept, the full main label is null and the development use is hold; `human_review.status=deferred` means the shared criteria ruling is not yet complete; it does not deny that the user has given opinions, nor does it overwrite the earlier agent suggestion. N22's formal negation by the user and the assistant's semantic judgement are clearly distinguished; the other 4 were not automatically presumed false in binary correctness merely because of supplementary opinions. Suggested completed answers are listed separately and marked as not model output, and gold is not rewritten.

The cumulative 47 finalized development inputs have been exported to [manifest v4](../configs/splits/answer_repair_nq_train_confirmed_dev_v4.json), the [C1 subset](../artifacts/answer_repair/nq_train_dev_v1/confirmed/through_batch04_c1.jsonl) and the [runtime subset](../artifacts/answer_repair/nq_train_dev_v1/confirmed/through_batch04_runtime.jsonl), containing 11 E targets and 36 correct controls; the 5 disagreements do not enter this subset. Original gold, frozen EM/F1 and C1 original byte content stay unchanged, and the calibration/reserved evaluation intersection is 0. More correct controls support checking harmful repair during development, but a selected development set still cannot prove overall net gain.

The latest cumulative web status file is the [fourth-batch import JSON](../artifacts/answer_repair/nq_train_dev_v1/review/NQ_train_pending_92.chat_confirmations_batch04_20261019.json): 39 confirmed, 22 pending, 31 blank on this page. It contains the opinions of earlier batches within the web page scope; back up other local drafts first, then import manually. The original 92-record material and its package hash are unchanged.

The remaining unreviewed are N51/N53/N54/N55/N58/N60/N61/N64/N65/N66/N68/N69/N70/N71/N72/N73/N75/N76/N77/N80/N83/N85/N86/N87/N88/N90/N92/N93/N95/N97/N100.

## Third batch of human opinions: 2026-10-18 (historical)

This batch strictly records only the 11 records N63/N67/N78/N79/N82/N84/N89/N91/N94/N98/N99. The original words and scope are in the [third-batch confirmation](../data/annotations/human/answer_repair_nq_train_dev_v1_batch03_20261018.confirmations.json), and the per-record conclusions are in the [third-batch human record](../data/annotations/human/answer_repair_nq_train_dev_v1_batch03_20261018.jsonl).

| ID | Recorded result |
|---|---|
| N63, N67, N78, N79, N82, N89, N91, N98 | The user agreed with the original deferral opinion; known facts and undetermined fields are kept, without automatic classification; for N82 it is known that the original answer did not answer the solid classification, but whether the evidence is sufficient is still undetermined |
| N84 | The user judged it correct; R1/R2 directly support Henrietta Maria, and R3 also lists it as the official naming account. Added to correct controls; the reference's alternative account does not override the original answer |
| N94 | The user judged it correct; the duty-summary granularity of this question is accepted, and it is added to correct controls. This individual case does not establish a rule that jurisdiction-scope qualifications can generally be omitted |
| N99 | The user confirmed that the content the question requires is not given in the retrieved evidence. R4 has the Back-to-Africa name but does not support the "began in the 1950s" relation. `evidence_sufficient=false` is confirmed separately; the reference's time/question-intent doubts remain, C1 correctness and the full main label stay deferred, the candidate direction is Insufficiency, and it is not included in the E core |

This batch adds 2 correct controls and no new utilisation targets; the other 9 stay deferred, of which N99's missing-evidence judgement has been recorded separately. The cumulative total is 11 E targets, 3 correct controls, 17 deferred and 69 unreviewed. The original 23 disputes have all received human opinions, resulting in 3 E targets, 3 correct controls and 17 deferred; this is review processing progress, not E model performance.

The [cumulative 14-record development manifest v3](../configs/splits/answer_repair_nq_train_confirmed_dev_v3.json), the [frozen C1 subset](../artifacts/answer_repair/nq_train_dev_v1/confirmed/through_batch03_c1.jsonl) and the [pure runtime subset](../artifacts/answer_repair/nq_train_dev_v1/confirmed/through_batch03_runtime.jsonl) have been exported, and the calibration/reserved intersections are both 0. N52/N84/N94 still keep the original EM=0, and human-reviewed semantic correctness is counted separately; 3 selected correct controls are still insufficient to reliably estimate the overall harmful repair rate. N99 does not enter this repairable subset; gold was not rewritten, no external facts were added, and no GPU was run.

The latest web status file is the [third-batch cumulative import JSON](../artifacts/answer_repair/nq_train_dev_v1/review/NQ_train_pending_92.chat_confirmations_batch03_20261018.json): 6 confirmed, 17 deferred, 69 blank on this page. The original 92-record package and its hash are unchanged. First export other local review backups, then import this file; it contains the second and third batches of opinions, does not read browser drafts, and does not automatically overwrite browser state.

## Second batch of human opinions: 2026-10-18 (historical)

This batch only records the 12 records the user explicitly mentioned. The full original words and per-record rulings are in the [second-batch confirmation](../data/annotations/human/answer_repair_nq_train_dev_v1_batch02_20261018.confirmations.json), and the per-record status is in the [second-batch human record](../data/annotations/human/answer_repair_nq_train_dev_v1_batch02_20261018.jsonl). Agreeing to defer means accepting deferred handling; correctness or the main label was not automatically filled in from null.

| ID | User ruling and implementation |
|---|---|
| N3, N7 | Agreed to defer: question intent and old/new names / buyer and purchase method unclear |
| N8 | Subject identification error; this case is understood as the song, R2 is Deep Blue Something, and C1's Audrey Hepburn corresponds to Moon River in the film; internally mapped to E utilisation |
| N19, N28 | Agreed to defer: the evidence relation and question intent/reference still have problems |
| N33 | R1 clearly explains the origin of the name; C1's location fact is supported, but it omits the origin from the sign with missing letters and the three members plus 3. `partial` is kept, internally mapped to E utilisation |
| N44 | Agreed to defer: relative time gives no starting point |
| N52 | The user confirmed the answer is correct; the time granularity of September 2017 is accepted for this case, as a correct control. The original text gives the month entry was announced, and this is not extended to mean that every announcement time equals the actual event time |
| N56, N57, N59 | Agreed to defer: medium, measurement dimension and ATP calculation criterion etc. still undecided |
| N62 | This case should answer the full name Lars Dittmann Mikkelsen; the main surname Mikkelsen is missing, recorded as `partial`, internally mapped to E utilisation |

This batch adds 3 E targets and 1 correct control, with another 8 explicitly deferred. The E classification for N8/N33/N62 is an internal mapping based on the user's confirmation of the specific errors and the frozen evidence, and does not pose as an official upstream Diagnosis label. The `initial_supported` of the four finalized records stays null, since the user did not separately confirm this score; for N33, the supported location fact and the incomplete answer are kept separately. The N62 ruling only applies to this case's missing key surname, and it cannot be inferred that all person names must output the full legal name.

The [cumulative 12-record development manifest v2](../configs/splits/answer_repair_nq_train_confirmed_dev_v2.json), the [frozen C1 subset](../artifacts/answer_repair/nq_train_dev_v1/confirmed/through_batch02_c1.jsonl) and the [pure runtime subset](../artifacts/answer_repair/nq_train_dev_v1/confirmed/through_batch02_runtime.jsonl) have been exported. All come from development, the intersection with calibration/reserved evaluation is 0, and the runner's frozen scoring precheck passes. N52's original EM=0 and gold are kept, and human-reviewed semantic correctness is counted separately. Only 1 correct control is still insufficient to reliably estimate the harmful repair rate; there are no new GPU results and no algorithm improvement conclusions.

The original web page keeps the same 92-record material and data package hash, to avoid invalidating existing browser exports. The page validation was updated to allow per-record human utilisation rulings for "partially correct but missing key content"; partially correct is not automatically classified as utilisation. The current web state is synced through the [import JSON of this round's opinions](../artifacts/answer_repair/nq_train_dev_v1/review/NQ_train_pending_92.chat_confirmations_20261018.json): 4 confirmed, 8 deferred, 80 blank. First reopen/refresh the updated HTML, then click "Import existing review". Importing replaces the page's local drafts; if the browser has other reviews, export a backup first. This file only reflects the opinions of this conversation and does not read or overwrite any browser draft.

## Assisted initial review groups

| Original screening group | Suggested semantically acceptable | utilisation candidates | Deferred | Total |
|---|---:|---:|---:|---:|
| EM=0 and reference passage hit | 49 | 8 | 23 | 80 |
| EM=1 controls | 20 | 0 | 0 | 20 |
| Total | 69 | 8 | 23 | 100 |

The table above keeps the grouping of the initial **agent suggestions**. After the first batch of 8 utilisation candidates was confirmed by the user, the remaining 92 entered web review; the second batch then gave the above opinions on 12 of them. The per-record assisted reasons are based on all C1 Top-5 and the offline reference, without querying external facts and without using C2/C3 or new E generation results. Citations all keep passage_id, the original character positions and the evidence hash, and all 100 records are in the internal development part. The original labels on main and the frozen scorer were not overwritten.

`initial_correct` indicates whether the requested fact was answered; `initial_supported` indicates whether the evidence can support it as the answer to this question, not whether the text appears. In ambiguous cases it may also be recorded that one interpretation is supported while correctness is kept null; see the reason for the specific interpretation. This assisted field alone cannot be treated as a faithfulness score of a statement detached from the question.

The 49 suggested acceptable in the screening group are commonly due to valid aliases, year vs. full date, additional city and country information, well-supported paraphrases, digits vs. English number words, and PTB vs. Unicode punctuation. Others involve answer completeness or factual qualification and need human ruling. The 69 include the 20 original EM-matched controls; 20→69 cannot be reported as E's improvement, and E has not yet been run on this batch.

This batch was deliberately screened 80:20 and comes only from the development part, so 8/80 or 69/100 cannot be extrapolated to NQ Train's class ratio, semantic accuracy or overall gain. The development judgement it supports is: **EM=0 with a gold passage hit only provides a candidate and cannot automatically define utilisation.**

## The 8 confirmed target errors of the first batch (historical)

The answer directions below are suggestions after the assistant read the existing evidence, not repair results already generated by E. The N numbers belong to `answer_repair_nq_train_dev_v1` and are separate from the old v7 review numbers.

| ID | Question and C1 | Answer direction given by the evidence | Suggested failure point |
|---|---|---|---|
| N6 | Harry in Harry and the Hendersons; Graham Greene | Kevin Peter Hall | Took the actor of a same-named character in Twilight; R1 clearly supports the target work |
| N29 | When did Han die; Han | Tokyo Drift | A person name does not answer the work in which the event occurs |
| N31 | Singer of The Most Beautiful Girl in the World; Charlie Rich | Prince | Confused similar but different song titles |
| N48 | India's largest man-made lake; Dhebar Lake | Govind Ballabh Pant Sagar | Ignored the comparison relation of second-largest / after |
| N49 | Who kissed the frog to turn it into a prince; Princess Emma | The princess in the modern Frog Prince story | In the cited adaptation Emma herself turns into a frog, a different transformation outcome; the assistant's original confidence was medium, and the user has confirmed this judgement |
| N74 | Who are the Twins in Matrix Reloaded; fictional characters | The Merovingian's henchmen | The category is too general and does not answer the specific identity |
| N81 | Number of episodes in True Blood season 7; 16 | 10 | Took the season 7 count of another series |
| N96 | Origin of the name monkey bread; Hungarian literature | Eaten by pulling apart by hand like a monkey | Treated the dish's spread history as the origin of the name |

The user has confirmed that the original answers of the above 8 are wrong and that the existing evidence is sufficient and clear; they are recorded as utilisation in E's human review layer. The label in the original main delivery is still null, and this does not pose as completion of full Diagnosis labelling; the answer directions in the table are still suggestions after human/assistant reading, not records of successful model repair.

The [8-record confirmed development manifest](../configs/splits/answer_repair_nq_train_confirmed_dev_v1.json) has been fixed, and the [C1 input](../artifacts/answer_repair/nq_train_dev_v1/confirmed/batch01_c1.jsonl) and [pure runtime input](../artifacts/answer_repair/nq_train_dev_v1/confirmed/batch01_runtime.jsonl) exported. Each record matches the original frozen C1, selected only from development, the overlap with calibration and reserved evaluation is 0, and the existing E runner interface check passes. This batch has only errors and no confirmed correct controls, so it cannot be used to estimate overall net gain or harmful repair rate.

## The 23 deferral suggestions of the initial assisted review (historical)

| ID | Deferral reason |
|---|---|
| N3 | The question is incomplete; the old/new name intent of Food Stamp Program vs. SNAP is unclear |
| N7 | purchased by may ask for the buyer or the purchase method |
| N8 | Ambiguity between the song Breakfast at Tiffany's and Moon River in the film |
| N19 | The composer/performer relation of Gran Torino and the Kyle/Clint reference are unclear; existing passages are insufficient to independently confirm the reference |
| N28 | The question asks for the real-person prototype, the reference gives the novel's protagonist, and the evidence does not state the prototype |
| N33 | Two readings of 3 Doors Down: naming location vs. name origin |
| N44 | The relative warming time lacks a starting date; historical warm-night records cannot answer a future starting point |
| N52 | The correspondence between the month entry was announced and the actual participation time/season needs confirmation |
| N56 | Boruto manga and anime release times differ, and the question does not specify the medium |
| N57 | bigger does not specify radius, mass or volume; 9 and 95 correspond to different dimensions |
| N59 | ATP stage yield vs. total yield and different measurement conventions conflict; the reference has a scope problem |
| N62 | The person name omits Mikkelsen; it cannot be treated as a valid alias just because it is a substring of the full name |
| N63 | The question's President and the reference's Prime Minister are mismatched offices; evidence for that year's president is missing |
| N67 | The incomplete question may ask for a definition or a discipline |
| N78 | Dinah and the Cheshire Cat are both in the related works, and the question does not specify which cat |
| N79 | Neither the full Scrabble set vs. the number in hand nor the language edition is specified |
| N82 | Answered the use rather than the solid classification; the explicit Top-5 support for molecular solid and the reference wording need confirmation |
| N84 | The name Maryland has official and alternative explanations; C1 has direct support, and the reference is not the only acceptable source |
| N89 | The numeric tolerance between over US$35 and US$35 and the price criterion the question refers to need confirmation |
| N91 | Whether closed system excludes external torque, and the completeness of the conditions for internal exchange and total conservation, need confirmation |
| N94 | Whether it is still an acceptable summary after deleting the jurisdiction scope of Commonwealth Agencies |
| N98 | Event ambiguity between starting production of the pilot and the official premiere |
| N99 | The question's 1950s time condition does not correspond to the reference's early intellectual source, and it is not a single movement |

Deferral is not a new main label and does not prove C1 is necessarily wrong. These records can continue to be set aside and do not block using the confirmed old development samples for engineering changes. The reference citation markers of N9/N18 and the reference doubts of N59/N63/N84/N99 only record problems; gold is not rewritten.

## Constraints on E's next development round

Actual strategy changes in the next round still use the v8 name; this round did not generate an unmodified v8 experiment package. Engineering can first advance with the existing human-confirmed development set, then gradually include the cases confirmed in this batch.

1. First check whether the original answer already answers the requested fact, keep it when support is sufficient, and reduce unnecessary re-answering. The check can only look at the question, the original answer and the runtime evidence.
2. When repair is needed, check whether the candidate matches the question in work, person, relation, time, measurement dimension, comparison and qualifiers; merely finding the quote verbatim is not enough.
3. Also keep the location of the quoted original text, to avoid changing to an untraceable answer; why/how and similar questions determine sufficient information by the requested fact, and cannot be uniformly and mechanically trimmed to a single name.
4. Ablate "check the original answer first" and "then check the candidate" separately, and compare with C1 and the existing v7 on the same development input batch. Record changed to correct, harmful repair, no change, parse failure and extra call cost.
5. Frozen EM/F1 continues to be reported separately, with user-confirmed semantic correctness and evidence support reported additionally. Gold cannot be modified nor the scorer relaxed to manufacture positive gain. The final shared test still needs coordination with the Diagnosis owner, and using the reserved evaluation part to select rules is forbidden.

These are development constraints to be verified and do not mean any checking model or strategy is already effective. First use a small batch to check behaviour changes and harmful repairs, then decide whether to expand GPU experiments.

## Storage and verification

- [92-record pending-review web page](../artifacts/answer_repair/nq_train_dev_v1/review/NQ_train_pending_92.html): reuses the layout of the old review page and can be opened offline. The 23 disputes are listed first, followed by the 69 suggested semantically acceptable; original N numbers are kept, and the 8 previously confirmed are excluded.
- [92-record pending-review full text](../data/annotations/assisted/answer_repair_nq_train_dev_v1_20261017.pending_92.full_evidence.txt): same scope as the web page, containing the original question, C1, reference, assisted reasons and all 460 evidence passages.
- [100-record readable judgements](../data/annotations/assisted/answer_repair_nq_train_dev_v1_20261017.txt)
- [100-record original question, C1, frozen reference and full Top-5](../data/annotations/assisted/answer_repair_nq_train_dev_v1_20261017.full_evidence.txt), listing the 8 target candidates first, then the 23 disputes, with N numbers unchanged
- [100-record structured judgements](../data/annotations/assisted/answer_repair_nq_train_dev_v1_20261017.jsonl)
- [Counts, priority numbers and source verification](../data/annotations/assisted/answer_repair_nq_train_dev_v1_20261017.summary.json)
- [Original frozen review material and integration method](ANSWER_REPAIR_NQ_TRAIN.md)

The completeness check of the initial assisted review passes: the 100 IDs match the frozen manifest order and are all in development; all original record hashes and verbatim citation positions match. The original review package, assistant judgements and original-text display files keep the frozen snapshot of the then-pending state; the current status follows the cumulative human review layer. The first-batch confirmation file strictly keeps the above 8 IDs, the second batch is saved separately, and the scope of the original 92-record web material is unchanged. The main delivery files, source shards, gold, frozen scorer and v7 results were not changed. Assisted record SHA-256: `920ea1c56f1b9512346e0d84fcd1d8e9e8e6abe9e62478e7ba2392378dece389`.

## Web review method

Open `NQ_train_pending_92.html` in a browser; no server or network is needed. On the left you can jump by original N number, search or filter; each record shows the original question, C1, the expandable frozen reference and the full Top-5, the judgement is filled in on the right, and the assisted reason can be expanded separately.

Human fields start empty. For the 69 samples suggested acceptable, clicking "Adopt assisted suggestion as draft" only fills in a draft; after checking, you still need to enter the reviewer and click "Confirm this human review". When a dispute cannot be ruled on, enter the reason and "Save as deferred"; deferred records are not counted as confirmed. Modifying a confirmed record returns it to draft.

Drafts are saved locally in the current browser, and browser restrictions or data clearing may lose them. After reviewing each batch, click "Export review JSON" at the top to back up; you can import it into the same version of the web page to continue reviewing. Importing verifies the review package, sample IDs, original N numbers and source hashes, then prompts to replace existing drafts. Clearing only affects this page's local drafts; the source files and the earlier 8 confirmations are unchanged.

After the export file is returned, it is checked and stored per explicit per-record human operations. The web page does not modify the repository automatically; the old v6/v7 review importers do not apply to this page's `nq_train_e_human_review_v1` format. The page, assisted suggestions and UI tests do not add any human confirmation.

Reproducible build: `.venv/bin/python scripts/build_nq_train_pending_review.py`. The builder first verifies the source of the frozen 100 records, assisted judgements and the first-batch 8 confirmations, and the original 92-record scope, then generates the HTML and a same-named manifest; the latter records the SHA-256 of the HTML, inputs and build source code. UI changes do not change the review data package hash, making it easy to keep importing existing drafts. 3 Python checks and 6 Node checks covering scope completeness, original text retention, import identity verification, confirmation conditions and draft state pass.

The preview checked display scope, evidence expansion, jumping, filtering, draft persistence, export and same-package import. Clicks in the last preview round did not respond, so the newly added in-page clear/overwrite confirmation dialogs did not complete browser re-verification; this item is not recorded as passed. UI tests only produced unconfirmed drafts and did not import any repository annotations.
