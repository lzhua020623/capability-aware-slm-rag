# E v7 experiment results: 2026-10-14 (Kaggle T4 x2)


The protocol is in [ANSWER_REPAIR_V7.md](ANSWER_REPAIR_V7.md), frozen before running. Result ZIP `answer_repair_v7_reviewed_outputs_export.zip` (SHA-256 `48e82e600f189b2b5e7f98063883ffa3b14574c26debfc50d011170f7308d5d1`), with the original 38 files in `artifacts/answer_repair/v7_reviewed/kaggle_20261014/`. 102 cloud tests passed and the 56 input files passed verification; locally, replaying all 541 records (536 unique IDs) with the frozen package code matches, see `release_audit/cpu_replay.json`. Environment: Python 3.13.15, torch 2.11.0+cu128, transformers 5.16.1, bitsandbytes 0.50.2, model versions fixed per protocol.

**Conclusion first**: FEVER's subject-entity constraint kept the offline replay gains on the development batches (reviewed 4→10, dev50 45→46, no harmful repair), but was net −3 on the 267 held-out records (1 repair / 4 harmful repairs); NQ's short-answer strategy was net negative or flat in frozen EM on all batches, significantly negative on held-out (71→62, McNemar p=0.0225), and the changes include real semantic harmful repairs. **Neither strategy is promoted.** Held-out has been run only once per protocol, and no further tuning will be done on these samples.

## Frozen metrics

| Batch | n | Initially correct | v6r1 final | v7 final | v7 repair / harmful repair | v7 changed answers |
|---|---:|---:|---:|---:|---:|---:|
| FEVER reviewed | 14 | 4 | 7 | 10 | 6 / 0 | 6 |
| FEVER dev50 | 50 | 45 | 44 | 46 | 1 / 0 | 1 |
| FEVER held-out | 267 | 232 | none | 229 | 1 / 4 | 5 |
| NQ reviewed (EM) | 14 | 2 | 0 | 3 | 1 / 0 | 2 |
| NQ dev50 (EM) | 50 | 20 | 7 | 19 | 2 / 3 | 9 |
| NQ held-out (EM) | 146 | 71 | none | 62 | 2 / 11 | 20 |

NQ F1: reviewed 0.292→0.368, dev50 0.549→0.512, held-out 0.632→0.586. The FEVER results on reviewed and dev50 are identical record by record to the offline replay of the design phase (same classifier scores, deterministic rules).

| Batch | Qwen calls / generation seconds / input tokens | NLI forward batches / classification seconds / input tokens |
|---|---|---|
| FEVER reviewed / dev50 / held-out | 0 | 29 / 12.6 / 21,910; 101 / 30.6 / 62,352; 537 / 170.1 / 337,180 |
| NQ reviewed / dev50 / held-out | 14 / 75.3 / 11,517; 50 / 293.4 / 45,385; 146 / 878.5 / 132,829 | 0 |

## FEVER `fever_nli_anchored`

The 267 held-out records by offline label: 232 correct (157 kept, 64 uncertain, 7 conflict, 4 repaired = 4 harmful repairs); 35 insufficiency (28 uncertain, 6 kept, 1 repaired = the only repair). This batch has no utilisation labels, so it can only measure "whether it breaks things": harmful repair rate 4/232 = 1.7%, abstention rate 92/267.

All 4 harmful repairs occurred on windows that passed the lexical subject constraint. Passing the constraint does not equal successful entity disambiguation: there are both cases where NLI ignored qualifiers/attribute categories and cases where the subject only captured sexual or city and county were not distinguished:

| ID | Claim | Decision | Reason |
|---|---|---|---|
| 49192 | The Armenian Genocide **rarely** was the extermination… | REFUTES→SUPPORTS (E=0.976) | NLI ignored rarely |
| 164986 | Polar bears are classified as endangered. | REFUTES→SUPPORTS (E=0.918) | Species conservation status misjudged |
| 14448 | Sexual slavery is a reason for human trafficking. | SUPPORTS→REFUTES (C=0.932) | The subject chunk is only "sexual", and the labor trafficking page was taken as a refutation |
| 126292 | Camden, New Jersey is a county. | REFUTES→SUPPORTS (E=0.984) | The Camden County page entails "is a county", city/county entity confusion |

The only repair, 104922 (Oscar Isaac / Apocalypse), cites the X-Men: Apocalypse page, and the window contains Oscar Isaac, so it is grounded.

## NQ `short_answer_quote`

Status distribution: reviewed 11 invalid / 2 repaired / 1 kept; dev50 32 / 9 / 9; held-out 99 invalid / 20 repaired / 25 kept / 2 unanchored. Of the 210 records, 142 are invalid, for these reasons: in 61 the model wrote the rank as `RANK: [1]`; 72 failed quote location; 8 quotes exceeded 60 words; 1 answer exceeded 20 words. Quote location failures include wrapping quotation marks, truncated trailing words (ridge/ridges), quoting the wrong passage (thumb twiddling quotes R3 but labels R1) and real sentence-deletion splicing (the Columbia case deleted ", Columbia 's 28th mission").

**Post-hoc replay with lenient parsing (not promoted, analysis only)**: `scripts/replay_short_answer_lenient.py` uses the same raw outputs, accepts `[n]`, strips wrapping quotation marks, and treats `ANSWER: INSUFFICIENT` as abstention; results are in `release_audit/lenient_parser_replay.json`:

| Batch | Frozen v7 EM | Lenient parsing EM | Repair / harmful repair after relaxing |
|---|---|---|---:|
| reviewed 14 | 2→3 | 2→3 | 1 / 0 |
| dev50 | 20→19 | 20→20 | 3 / 3 |
| held-out | 71→62 | 71→58 | 4 / 17 |

In this post-hoc replay, lenient parsing increased valid outputs but also increased harmful repairs, showing that fixing format alone is not enough to restore net gain. The model cannot see the initial answer, and re-selecting an answer may cause unnecessary replacements; this is still a causal hypothesis awaiting ablation, and the current comparison alone cannot establish that making the initial answer visible is necessarily safer.

The 146 held-out records by offline label: of 126 correct, 11 EM harmful repairs (84 invalid, 23 kept, 17 repaired, 2 unanchored); all 7 accessibility invalid; 13 insufficiency had 3 changed and none became correct; of the 55 "labelled correct but judged wrong by EM" samples only 6 were changed and 2 became EM correct. This batch has no utilisation labels, so it cannot reliably estimate the repair rate of the target error layer and is mainly used to check misrouting risk; lacking this label does not prove there are no real utilisation errors.

The assistant's initial reading of the 20 held-out changes (pending human review, see `release_audit/review_priority.md`):
- Real harmful repairs (wrong answer type or object): spirits etymology Middle Eastern alchemy→"spirit of the original material"; GoT grandmother actress Rigg→character name Lady Olenna Tyrell; South African Youth Day 1976→"16 June"; Only Love Can Break Your Heart's Neil Young→cover artist Saint Etienne; curling 1998→1924 is disputed.
- EM false regressions (content not wrong, only longer or with a preposition/prefix added): Himadri Station→Himadri, "On September 9, 2012", the space in Road/Track, "in southern Turkey", "off Flamborough Head", Paige O'Hara with a prefix, a more specific location for The Crossing, a more specific heart position.
- EM repairs: SA node→the sinoatrial node, 7→seven (both spelling alignment, not semantic repair).

Of the 9 changes in dev50, Greenland→"an autonomous constituent country within the Kingdom of Denmark" (asks for the continent) and Feed Jake→a lyric fragment (asks for the original singer) are real harmful repairs that do not answer the question; War of 1812 lists both sides; the rest are EM false regressions or false repairs. In the 14 reviewed records, the Annie singer is a real repair (already confirmed in v6r1), and the bias definition was only turned into a full sentence.

## What this round showed

1. FEVER: the subject constraint got 6/0 and 1/0 on the development batches, but the labelled harmful repair rate on held-out is 4/232 = 1.7%; this is the harmful repair proportion of the whole E decision chain on initially correct answers, not NLI's general error rate on entity-correct windows. The frozen Controller does not route FEVER to E on the PRIMARY test; FEVER single-module results are still used for Always-Verify and the failure type × operation matrix, and cannot substitute for adaptive path gains.
2. NQ: the candidate-first short-answer strategy is net negative in frozen EM on the first batch of E-unseen data, and the assistant's initial reading finds real semantic harmful repairs. The 20→20 of v4 dev50 came from grounded_reanswer / reanswer; grounded_reanswer likewise hides the initial answer and cannot serve as proof that verify-first is safer. The next round can test "check the original answer first, then minimal repair", but it must be ablated separately. New tests also need unused target errors added, and the repair rate cannot be estimated solely from the remaining 1,287 correct/insufficiency samples.
3. The `heldout3216` in the next section has completed one run, providing a repair signal at the utilisation label layer, but it cannot be treated as an upper bound of a reliable repair rate. It is stratified by label, the labels and semantic outputs still need checking, and it does not represent the deployment distribution.

## Human review queue

`release_audit/review_priority.md`: 43 changed results first (18 frozen metric regressions, 13 improvements, 12 other changes), with the remaining 498 kept answers. The 7 completeness criteria of the confirmation list are reused; the assistant's initial reading is not human ground truth.

## Batch 4 `heldout3216`: NQ held-out with utilisation labels (run only once)

Result ZIP `answer_repair_v7_heldout3216_outputs_export.zip` (SHA-256 `91ee085bb38c8ee05220b71e5b547978871d84635261d8034a177370a8b927fd`), with the original 11 files in `artifacts/answer_repair/v7_heldout3216/kaggle_20261014/`; the 50 input files passed verification, 102 cloud tests passed, and the local frozen code replay of 300 records matches. 300 Qwen calls, 1618.4 seconds of generation, 272,962 input tokens, an estimated 18,980 output tokens.

This is the only batch that E has never seen and that contains repairable samples (selected stratified by offline label, not representative of the deployment distribution). Overall frozen EM 63→67 (13 repairs / 9 harmful repairs, McNemar p=0.52), F1 0.304→0.334; by layer:

| Layer | n | EM before→after | Repair / harmful repair | Changed answers | invalid | F1 before→after |
|---|---:|---:|---:|---:|---:|---:|
| evidence_utilisation | 120 | 0→10 | 10 / 0 | 24 | 83 | 0.063→0.146 |
| evidence_accessibility | 35 | 0→3 | 3 / 0 | 9 | 18 | 0.076→0.164 |
| correct | 100 | 63→54 | 0 / 9 | 14 | 56 | 0.801→0.759 |
| evidence_insufficiency | 45 | 0→0 | 0 / 0 | 5 | 35 | 0.018→0.023 |

The 10 EM repairs in the utilisation layer (Tamara Drasin, Billy Idol, Davos, Astros 2017, Hasmukh Adhia, Planet of the Apes July 14, Brexit referendum June 23, AGT season 13, Red Guards 1966–1967, the cross 4th century) all replace a wrong entity/date with the correct value in the evidence; there are also cases with EM=0 but clear improvement, such as elevator pitch (F1 0.09→0.90), the Disneyland location (→0.89), Bastard Executioner "one season" and the poverty rate 13.5%; under the assistant's initial reading the semantic repairs in the utilisation layer are about 13–14/120. There are also changes in the wrong direction: Spock hand gesture→Jewish priestly blessing (the etymology in the evidence, while the reference wants Nimoy), hairpin, criminal law, Home Alone 2 New York→New York City (reference Florida) are all wrong→wrong and not counted as harmful repairs.

Of the 9 EM regressions in the correct layer, the assistant's initial reading considers 7 to be added words/added dates/spaces ("Hal ( Bryan Cranston )", "Burj Khalifa in Dubai…", "March 18, 2005", "the volume of a gas", "more in the wintertime", "1980 summer Olympics", "pulse - width"), 1 disputed ("RMS Titanic"→"Titanic", the question asks for the full name), and 1 a real harmful repair candidate (the Philippines' oldest street Colon Street→Philadelphia's Elfreth's Alley, entity error). 63 is the initial EM-correct count, not a human-confirmed semantically correct count; the formal semantic harmful repair rate requires reviewing initial/final semantic correctness and then re-determining the numerator and denominator.

Post-hoc replay with lenient parsing (`release_audit/lenient_parser_replay.json`, not promoted): overall still 63→67 (16 repairs / 12 harmful repairs); utilisation 0→11, correct 63→53. Consistent with the main package held-out: relaxing format only gains 1 more repair at the cost of 3 more EM harmful repairs. Of the 192 invalid outputs, 126 are non-verbatim quotes, 53 are `RANK: [n]` and 13 are too long.

**What this batch changes**: by the upstream utilisation labels, v7 has a frozen EM repair of 10/120 = 8.3%; the assistant's initial reading also finds some improvements not recognised by EM, but the semantic figure of about 11% is still an estimate awaiting human verification. The harmful repair candidates in the other held-out batch, such as actor→character, original singer→cover artist and year→date, show that protection of correct answers still needs improvement. The initial EM-correct counts of the two batches, 71+63=134, cannot be used directly as the denominator of the semantic harmful repair rate, nor can the assistant's estimate be written as a formal human metric. Combined with the misrouting risk, it is currently still **not promoted**. The next version tests checking with the original answer visible and minimal modification; the remaining 1,287 can serve as safety material, but are insufficient to independently accept the target repair capability.

Human review queue: `artifacts/answer_repair/v7_heldout3216/release_audit/review_priority.md`, 52 changes first (9 EM regressions, 13 improvements, 30 other).

## Result re-check, data decisions and next-round roadmap (2026-10-14)

This section re-checks the two batch archives at `5295edc`. The 38 files of the main package ZIP are byte-identical to the unpacked files in the repository, and the frozen code replays 541 records; the 11 files of the batch 4 ZIP also match, replaying 300 records. In total 841 records, 836 unique dataset/sample_id, and the 95 changed records correspond to 93 unique samples. pair_review of both batches is all pending, and the CPU replays both explicitly state semantic_correctness_verified=false. Model inference was not re-executed for the main package or batch 4.

### Where the current shortcomings come from

- Method: after NQ selects a new answer from scratch, there is no acceptance check of "why the new answer is more credible than the original"; quote provenance and weak lexical relevance are insufficient to guarantee the answer's target, event, time, answer type and completeness. FEVER's capitalised subject rule still extracts incomplete subjects and cannot distinguish Camden city/county; the classifier also ignores qualifiers such as rarely.
- Engineering: in the main package NQ 142/210 invalid, in batch 4 192/300 invalid (64%). Format repair is worth doing, but post-hoc lenient parsing did not eliminate the net loss, so it cannot be treated as the sole root cause.
- Data composition: the first held-out batch has no utilisation labels; it can reveal harmful changes but cannot estimate the target layer's repair rate. Only batch 4 added 120 utilisation-labelled samples.
- Label quality: the upstream annotation_source of all 300 records in batch 4 is ai_initial_nq3216_2026-10-09. needs_manual_review=false cannot be interpreted as human acceptance. The source consistency of the workbook gold/EM has been checked, but this does not guarantee that every reference answer and failure type is correct.
- Evaluation independence: the two held-out batches were data E had not used at the time, but still come from Diagnosis's training range; they cannot be called an independent test of the whole Diagnosis→Controller→E system. The results have now been read, and when changing strategies later these batches should be classed as seen analysis material.

### Expand, augment or replace

It is recommended to keep NQ/FEVER as the main tasks and to expand in a targeted way with verified target errors, correct controls and hard boundaries. More evaluation records do not automatically change the current non-fine-tuned Qwen/NLI weights; they mainly improve error analysis, calibration and statistical credibility. There is no evidence supporting replacing the main dataset to obtain positive results.

Re-checking upstream v2-expanded TRAIN, after excluding preliminary500 there are 1,587 NQ IDs in total, of which this round selected 300; the remaining 1,287 by existing labels are correct 1,164 and insufficiency 123, with **utilisation/accessibility both 0**. All 120 utilisation and 35 accessibility have been used in this round, so the remaining pool cannot alone serve as the next round's repair rate test. The same C1 conditions need to be rebuilt from new NQ/FEVER IDs not yet used for E development and the failure types verified, or the team should plan a brand-new shared test set in advance. First freeze development, calibration and test grouped by ID/source, then look at test results.

Data augmentation can be used for development stress tests, or when training a checker/acceptance policy is explicitly planned in the future: same-name entities, actor/character, original singer/cover artist, event year/report date, negation and qualifiers, quantity and range perturbations. Each synthetic label needs confirmation, partitioning is grouped by original source, and synthetic samples cannot pose as independent natural test samples.

An optional external supplement is [RAGTruth](https://github.com/ParticleMedia/RAGTruth): it officially provides multi-model RAG answers with word-level human hallucination annotations, suitable for additional testing of the checker. Its model outputs and tasks differ from this project's fixed-Qwen C1, so it cannot directly replace the end-to-end main evaluation. If the checker itself was trained on RAGTruth, the official test range not used for training/tuning should be used, and the known training sources should be stated.

### Next-round candidate roadmap: unified pipeline, controlled comparison, pending re-development verification

Both tasks use the same logic: question/claim + existing C1 + initial answer → check the original answer → minimal repair by the fixed Qwen when necessary → check the candidate and answer completeness → accept or fall back. The input wording and output metrics of FEVER and NQ are adapted separately; the checker is a replaceable component, to avoid using "the dataset changed" and "the model changed" at the same time as the basis for effectiveness. The v7 direct NLI label decision is kept as a historical baseline, and the extra model budget is recorded separately.

1. First confirm this round's key outputs and stratified labels. Prioritise the 93 unique changed cases, then sample unchanged/invalid outputs; do not review only EM regressions. Separately label initial/final correctness, evidence sufficiency, citation support and question match, keeping reference problems and boundary reasons.
2. Build a correct/incorrect and boundary checking set on the seen development material. First compare the false positive/false negative rates on original answers and the cost of the current MNLI, the proposal candidate `cross-encoder/nli-deberta-v3-small` and MiniCheck, and calibrate the abstention interval. The threshold 0.90 should not be copied directly across models.
3. Run the minimal set of ablations: no answer change, v7 candidate-first, repair after checking with the original answer visible, and the same repair scheme plus an independent check. Inputs, fixed Qwen, inference settings and cost criteria are consistent, only one factor is changed at a time; format parsing and semantic acceptance rules are evaluated separately. Making the original answer visible may anchor old errors, so missed repairs must also be reported.
4. Replace only when the check clearly identifies a problem with the original answer, the candidate fully answers the query and the evidence is more sufficient. Identical entity words cannot substitute for entity and relation matching; unsupported does not equal refuted, and a binary checker's unsupported cannot be directly mapped to FEVER REFUTES.
5. After freezing the selected method, run once each on new target errors and correct controls, then the team measures net gain, harmful repairs, coverage and full cost under Controller selection on a shared unseen test set. E's actual gain depends on the proportion of repairable errors entering E and the misrouting proportion of correct samples, and cannot be extrapolated from the overall EM of the stratified 300 records.

Sample size is determined by the goal: under the assumption of independent binomial sampling, if 300 confirmed correct controls have zero harmful repairs, the one-sided 95% upper bound of the true harmful repair rate is only about 1% (1−0.05^(1/300)). This only illustrates that rare harmful repairs need many correct controls; it is not a guarantee the current experiment has reached; repair differences still need power planning based on the expected paired improvement.

### Implementation references searched

| Implementation | What can be borrowed | Usage advice and limitations for this project |
|---|---|---|
| [MiniCheck](https://github.com/Liyan06/MiniCheck) | Sentence-level document-claim support judgement, with a public Flan-T5-Large version under 1B and a synthetic data method | Prioritise as a checker comparison; its output is supported/unsupported, cannot be directly three-way, and cannot automatically guarantee that the candidate answered the original question. See the [official model card](https://huggingface.co/lytang/MiniCheck-Flan-T5-Large). |
| [LettuceDetect](https://github.com/KRLabsOrg/LettuceDetect) | Takes question/context/answer as input and locates unsupported spans in the answer | Can be used for minimal repair after localisation; question match, answer completeness and missed detections still need separate measurement, and having no hallucinated span cannot be directly equated with answering correctly. |
| [RARR](https://github.com/anthonywchen/RARR) | The repair idea of editing the original answer only after an Agreement Gate, preserving the original content as much as possible | Prioritise as a reference for the gate and editing flow; the original implementation includes extra search and API models, so this project should implement a controlled variant under fixed C1/fixed Qwen and record all costs. |
| [AlignScore](https://github.com/yuh-zha/AlignScore) | Text-pair factual consistency scoring and chunked checking | Can serve as a second-stage scoring comparison; the official implementation's dependencies are old, so the environment should be isolated and the training sources checked, and paper leaderboard numbers are not to be taken as this project's results. |

The above are candidates and design references after checking the official repositories/model cards; they have not been tested on this project's GPU or promoted. The first round only needs the current checker, one lightweight NLI and one document-level checker as comparisons, and whether to replace is decided by harmful repair risk, repair coverage and cost.
