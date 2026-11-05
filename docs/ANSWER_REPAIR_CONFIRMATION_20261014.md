# E v6r1 key human confirmation checklist (2026-10-14)

> **Status: confirmed (2026-10-14, user replied "agree").** A and B are confirmed per the two-reading conclusions; C is re-judged per the recommendations below (113, 121 deferred); D is handled as recorded. Records are in `data/annotations/human/answer_repair_v6r1_confirmations_20261014.json` and `answer_repair_v6r1_pairs_20261014.jsonl`; after deduplication, 9 semantic repairs, 3 harmful repairs, 2 deferred. The remaining 97 unique samples are pending batch confirmation.

For the E owner to confirm item by item. Numbering follows 1–128 in `artifacts/answer_repair/v6r1_reviewed/assisted_review_20261014/assisted_review.md`. Each item gives two independent readings: Codex assisted review (2026-10-14, reference visible) and an independent blind review (Claude, 2026-10-14, seeing only the question, initial/final answers, the model's actual citations, the Top-5 original text and the frozen reference, **without seeing the Codex conclusions**). Neither reading is human ground truth; after confirmation, use `scripts/apply_answer_repair_confirmations.py` to write the human records, then aggregate semantic repairs/harmful repairs.

On the four boolean fields of the 28 key cases, the two readings fully agree on 22; all 6 differences fall on fields Codex marked "pending" (where the blind review gave a definite judgement), plus "whether the initial answer has support" for #30 (whether a reasonable refusal counts as having support). Not a single item has the two readings in opposite directions.

## Unified answer-completeness criteria (recommended for adoption)

1. **Correct** = answers the object, relation, time and scope the question asks for. Synonymous short answers and long sentences that contain the correct value without wrong claims both count as correct; long sentences are not judged wrong because EM=0, and short answers are not judged wrong because the reference is longer.
2. **Definition questions** (what is X / what does X mean) must give the defining predicate; merely repeating the name does not count as correct; omitting a minor qualifier (e.g. measurable) still counts as correct, but record `completeness_boundary`.
3. **who/when/where** must give a concrete entity/date/place; giving only a relative position or a related fact does not count as correct.
4. **latest/last/originally** are judged by the explicitly dated target event in the frozen corpus, not by current knowledge.
5. **Refusals** (not specified, etc.) do not count as correct answers; when evidence is genuinely missing, record it as a "grounded refusal", but count it as not repaired.
6. **FEVER**: SUPPORTS requires the whole claim to be derivable from the evidence, REFUTES requires the evidence to contradict the claim; when the evidence is silent, tentatively follow the frozen label and mark `reference_dependent`, which does not count as independent factual confirmation.
7. **Supportiveness** is judged separately from correctness: a true sentence that does not answer the question does not count as support; whether the model's actual citations and the Top-5 are sufficient is also recorded separately.

## A. Repair candidates (9 unique samples): recommend confirming all as semantic repairs

| No. | Dataset / ID | Question | Initial → Final | Codex | Blind review | Notes |
|---|---|---|---|---|---|---|
| 18 | FEVER 156493 | Daggering is not a type of dance. | SUPPORTS → REFUTES | Repair | Repair | R1 first sentence "is a form of dance" |
| 19 (=21) | FEVER 203615 | Sugarland Express co-written by Spielberg | REFUTES → SUPPORTS | Repair | Repair | Duplicated across two batches, counted only once |
| 20 | FEVER 72578 | Greek has speakers in Albania | REFUTES → SUPPORTS | Repair | Repair | |
| 22 (=33) | NQ tibia/fibula | where is the tibia and fibula bone located | on the lateral side → "The fibula ... is a leg bone located on the lateral side of the tibia ..." | Repair (medium confidence) | Repair (medium confidence, the tibia's location is only implicit) | The only one needing your call: per criterion 3, the initial answer giving only a relative position is wrong; the final answer gives leg. Recommend counting as a repair |
| 23 (=35) | NQ Annie | who sang oh annie i not your daddy | song title → full sentence with Kid Creole and the Coconuts | Repair | Repair | Missed by EM |
| 25 (=38) | NQ Loihi | what is loihi and where does it originate from | name → submarine volcano + southeast of Hawaii | Repair | Repair | Criterion 2 |
| 27 | NQ serrated adenoma | what is a serrated adenoma of the colon | disease name → precancerous flat lesion of the colon | Repair | Repair | Criterion 2 |
| 28 | NQ Astros | who did the astros play last time they were in the world series | White Sox (2005) → Dodgers (2017) | Repair | Repair | Criterion 4 |
| 29 | NQ Jenny | when does jenny come back to gossip girl | report date → sixth-season finale | Repair | Repair | The initial answer took the report date as the return time |

## B. Harmful repair candidates (3): recommend confirming all as semantic harmful repairs

| No. | Dataset / ID | Question | Initial → Final | Codex | Blind review | Model citation |
|---|---|---|---|---|---|---|
| 1 | FEVER 140820 | Men in Black II is a film from America. | SUPPORTS → REFUTES | Harmful repair | Harmful repair | Cites the 2015 Belgian film Black, wrong entity |
| 2 | FEVER 33982 | Henry Cavill starred in two Superman films. | SUPPORTS → REFUTES | Harmful repair | Harmful repair | Cites Henry's Crime, wrong entity |
| 42 | NQ statute | a legislative act passed by congress is an example of | statute → "The act was passed by large majorities in both houses of Congress." | Harmful repair | Harmful repair | The citation holds verbatim but does not answer the question; frozen EM is 0 both before and after, missed by the metric |

Under v7's subject-entity constraint, neither FEVER harmful repair is triggered any more in offline replay (see `docs/ANSWER_REPAIR_V7.md`).

## C. Unresolved items (7): recommended default handling

| No. | Question | Point of dispute | Codex | Blind review | Recommendation |
|---|---|---|---|---|---|
| 31 | what is property of a system in thermodynamics | Whether the initial answer omitting "measurable" still counts as correct | Initial pending | Initial correct | Per criterion 2 judge **initial correct**; correct both before and after, not counted as a repair |
| 32 | what is the basic principle of christian morality | Whether the single word "love" is sufficient | Initial pending | Initial correct (criterion boundary) | Judge **initial correct**, not counted as a repair; record completeness_boundary |
| 54 | The Mighty Ducks was only produced by a Canadian film producer. | Whether C1 is sufficient to refute; the citation is the sequel D3 | Support pending | Support = yes (low confidence, partial) | REFUTES correct before and after; record supportiveness as **insufficient** (cites the sequel, producer nationality missing), does not affect the repair count |
| 109 | what does hp mean in war and order | Evidence gives only the general meaning and does not mention this game | Support pending | Support = yes | Correct before and after; judge supportiveness **yes** (the general definition suffices to answer the abbreviation's meaning) |
| 113 | little league pitching distance | Top-5 has only MLB's 18.44 m, and the reference itself is also the MLB value | Correctness pending | Correctness null | **Deferred**, mark reference_mismatch; not counted as repair/harmful repair |
| 121 | when did the soviet union start and end | Reference "in Eurasia" does not answer the question; 1991 is not in the evidence | Correctness pending | Correctness null | **Deferred**, mark reference_mismatch; not counted |
| 125 | who sings i just want to use your love tonight | The evidence lacks this lyric; requires lyric → song-title inference | Support pending | Support = yes | Correct before and after; judge supportiveness **yes** (matching the song title Your Love to the lyric is a routine inference) |

## D. Reference mismatches and criteria cases: recommended handling

| No. | Question | Agreed conclusion of both readings | Recommendation |
|---|---|---|---|
| 30 | what degree is a crock pot on low | Wrong before and after; reference 71–74 °C corresponds to the warming setting; the final answer is an irrelevant sentence | Record reference_mismatch + nonresponsive_final; the initial refusal does not count as correct per criterion 5. The blind review considers the initial answer unsupported (a refusal does not count as support), Codex considers it grounded; simply record it as "refusal grounded but not answered" |
| 41 | Which festival the church used to replace Samhain | Wrong before and after; the final answer cites the December 25 passage, wrong event; the correct value is not in the Top-5 | Record wrong_event_citation, not counted as a repair |
| 36 | UNC's last absence from the NCAA | Wrong before and after; the final answer talks about a venue boycott, wrong event | Same as above |
| 101 | paleomagnetism ... as demonstrated by | Both readings judge wrong before and after (the answer restates the premise and gives no rock-layer record) | Keep your earlier correction rationale for old item 17 |
| 103 | Pharmacy school white coat | Both readings judge correct before and after; reference "before the first year" refers generally to medical school | Keep the confirmation of old item 23 |
| 104 | who was originally cast to play indiana jones | Both readings judge wrong before and after (evidence clearly states Tom Selleck was chosen, Ford was only suggested); evidence clear but not repaired | Record missed_repair; this is a case for v7 to focus on |
| 3 / 24 / 26 | Ringo Starr / bias / Columbia | Correct before and after; the final answer is only a grounded expansion | As representative cases of spurious EM regression; the remaining 12 NQ EM regressions are the same (items 5–17) |

## E. What you need to reply

Just reply by number, for example:

> A all agree; B all agree; C: 31, 32 initial counts as correct; 54 support insufficient; 109, 125 support yes; 113, 121 deferred; D agree.

After replying, run:

```sh
python3.11 scripts/apply_answer_repair_confirmations.py --confirmations <your_confirmations.json> --out data/annotations/human/answer_repair_v6r1_pairs_20261014.jsonl
```

It writes the confirmations into each record's `pair_review` (including duplicate records), binds the result SHA-256, and outputs per-batch semantic repair/harmful repair counts. Per the recommendations above, the deduplicated v6r1 semantic result will be: **9 repairs, 3 harmful repairs, 2 deferred**; the remaining 109 items keep the assisted review conclusions, pending later batch confirmation.
