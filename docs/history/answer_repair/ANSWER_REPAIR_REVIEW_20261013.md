# E development sample assisted review (2026-10-13)

This batch has completed **37 Codex assisted review opinions**; the human-confirmed count when the package was issued on October 13 was **0**. Each record keeps the original question, the initial C1 answer, the complete Top-5 in original order, the original diagnosis label, the offline reference answer, evidence excerpts, a suggestion, and one question pending human confirmation. Assisted opinions are not human labels, independent verifier outputs, or new experimental effects.

Progress as of 2026-10-14: cumulatively **37/37 processing opinions confirmed**. Records 1–10, 16–18, 20, 24, 26, 28, 31, 32 are included as E utilisation core development candidates (19 records); records 11–14, 23, 33, 34, 36, 37 are kept as correct controls (9 records); records 15, 19, 21, 22, 25, 27, 29, 30, 35 are deferred from inclusion and saved separately as ambiguity/label-boundary cases (9 records, labels still unresolved). 28 candidates in total are usable for the next development round; all processing opinions being confirmed does not mean all label disputes are resolved. Record 17 was changed from the agent's deferral suggestion to utilisation after the user supplied the failure reason; the user's original explanation and the agent's qualification of the wording are recorded separately. Record 23 is kept as a correct control, and the scope qualification "most pharmacy students at the end of the first year" (original wording "大多数药学院学生在第一学年结束时") is recorded per the user's suggestion, without rewriting the original answer. Record 33 keeps a note on associative inference; record 35 is deferred as a correct control per the suggestion revised in chat, because "appearing as a hint" is not the same as "explicitly revealed". Records 36–37 were explicitly supported by the user on their own initiative and are recorded as confirmation of the existing correct-control suggestions, without claiming that a separate final group table had been shown before. Records are in the [chat confirmation file](../../../data/annotations/human/answer_repair_review_20261014.jsonl), which keeps the user's original replies and the corresponding confirmation scope. The review method was chat-assisted confirmation and cannot be described as independent record-by-record source re-checking or blind review. The original review package is unchanged; there are also [review results importable into the compatible page](../../../artifacts/answer_repair/review_20261013/E_human_reviews_chat_20261014.json).

Open the [local review page](../../../artifacts/answer_repair/review_20261013/E_assisted_review.html) to start; no GPU, server, or network is needed. Source diagnosis labels, gold, and previous-round experimental results have not been rewritten.

The 28 reviewed records have been used to prepare the [v5 development experiment](ANSWER_REPAIR_V5.md), and the list is saved separately as the [fixed development manifest](../../../configs/splits/answer_repair_reviewed_dev_v1.json). The packet inside the original page still keeps its state at package issue; to view chat confirmation results, import the review JSON above. Human re-review of outputs will be done separately after the real model run and will not be mixed into this initial sample review.

## Sample scope

| Dataset | utilisation candidates in training intersection | Already used in previous dev50 | Newly included utilisation this round | New correct controls |
|---|---:|---:|---:|---:|
| FEVER | 10 | 1 | 9 | 5 |
| NQ | 17 | 4 | 13 | 5 |
| Total | 27 | 5 | 22 | 10 |

From the 500 frozen C1 samples of each dataset, take the training-set intersection of Diagnosis v1/v2, exclude all validation/test IDs of both versions, and require the original `needs_manual_review=false`. This covers all utilisation-labelled samples in that scope. The 5 correct controls per dataset are selected from eligible samples never in the previous dev50, by `SHA256(E-review-control:42:dataset:sample_id)`; controls were not picked based on this review's results.

In total 32 are newly included this round and 5 were already used for E development; samples viewed in this review subsequently count as development material. The NQ 3216 expansion pool is not in this batch's scope; this batch keeps the same C1 source as the previous round. A cohort selected by targeting labels does not represent the overall class distribution and cannot be used to compute overall deployment gains.

- C1 commit: `6f08f2fe2dc2dcf39b75177763207682b2890a31`
- Label and split commit: `7c7863f98f87e892119481a79e23ca6dcebdced0`
- Full source paths, hashes, and selected IDs: [review_packet.json](../../../artifacts/answer_repair/review_20261013/review_packet.json)

## Assisted review conclusions

| Original sample group | Codex suggestion | Count |
|---|---|---:|
| utilisation candidates | Can be submitted for human confirmation; suggested for inclusion as E core development candidates | 18 (FEVER 10, NQ 8) |
| utilisation candidates | Label boundary, reference answer, or question wording needs further re-review; deferred from inclusion | 8 |
| utilisation candidates | The original answer already has targeted evidence support; suggest checking the original label | 1 |
| correct controls | Preliminarily supported for keeping, still pending human confirmation | 9 |
| correct controls | Entity ambiguity; suggest deferring as a clean control | 1 |

It is suggested to first select "Disputed items suggested for priority review" on the page, 10 records in total. The following cases most affect the validity of the cohort:

| ID | Doubt | Please confirm by human review |
|---|---|---|
| NQ `-5471777320348063942` (pharmacy white coat) | The original answer "end of the first year" is supported by R5's pharmacy-specific description; gold is taken from R1's generic "some schools" description. | Was the original answer actually correct? Suggest moving it out of the error core set, but do not automatically change the source label. |
| NQ `-3112358457025326292` (No Game No Life) | Gold's sixth volume corresponds to the film adaptation material; the question seems to ask about the end point of the TV anime. | First clarify what is being asked, and confirm whether the Top-5 really contain evidence of the adaptation end point. |
| NQ `6680816286694593040` (Patriots) | R1 says ten times, R3/R5 say nine times; corpus time points are mixed. | What are the cutoff date and the rule for handling conflicting evidence? Can it count as utilisation under clear evidence? |
| FEVER `55426` (Firefox control) | R2 is a game with the same name, R5 is the browser; the claim is not disambiguated. | Does the original REFUTES hold under the current input? It cannot be deemed a clean control just from the correct label. |
| NQ `-288937641151618309` / `-5670674709553776773` / `678930299284183680` | The original answer equals a value in an unrelated show/project passage, and the correct fact is also in the Top-5. | How should the team distinguish accessibility caused by distraction from utilisation? A single output cannot prove causation. |

There are also answer-granularity issues for paleomagnetism and Easter lily, and the WWI question provides no options, making the answer non-unique. Among the controls, Gothic art and Smoking Man also keep human confirmation questions about predicate strength; the original labels are not treated as ground truth exempt from review.

All suggestions are judged only against the frozen C1 corpus; no current web knowledge was used to supplement evidence, and C3/70B results were not used as a label basis. Dynamic questions are interpreted at the corpus time point. The 52 quoted passages of original evidence have been verified verbatim; this only proves the source of the excerpts and cannot replace human semantic judgement.

## How to review

1. Select a sample on the page and first read the question, the original answer, and the necessary Top-5; long evidence can be expanded one by one or with "Expand all evidence".
2. Expand the Codex assisted opinion to view the located source text and doubts. The original label and gold are in a separate collapsed area, to avoid simply following the reference answer from the start.
3. Fill in the reviewer, original answer status, evidence status, suggested label, and usage. Assisted opinions are not automatically filled into the human fields.
4. Click "Confirm this human review". Inclusion in the E core requires confirming all of: the original answer has a substantive error, the evidence is sufficient and clear, and the label is utilisation. When you cannot decide yet, choose disputed/deferred and record the issue.
5. After finishing each batch, click "Export review JSON". The browser saves a local draft, and the exported file can be re-imported into the page of the same version. Always export before switching browsers or review package versions.

This page is not a blind review tool: old label categories are visible and assisted rationales can be expanded; this kind of assisted review cannot be described as independent double-blind annotation. Human disputes should be returned to the Diagnosis owner for adjudication under shared rules before building a new E cohort. Final IDs, label basis, and scoring criteria must be frozen before formal controlled experiments.

## Reproduction and validation

```sh
python3.11 scripts/prepare_answer_repair_review.py
```

The [preparation script](../../../scripts/prepare_answer_repair_review.py) reads fixed Git commits, the previous round's manifest, and the [separately saved assisted opinions](../../../data/annotations/assisted/answer_repair_review_20261013.jsonl). Output is written to the Git-ignored `artifacts/answer_repair/review_20261013/`, and does not modify source annotations or run the model.

Verified: 37 unique IDs, 185 original evidence passages fully preserved, zero overlap between the held-out splits of both versions, 37 agent-only opinions, 52 excerpts matched verbatim, HTML embedded data consistent with JSON. The page was opened and checked in Chrome; the human confirmation logic was checked for missing fields, wrong citation numbers, and inconsistent inclusion conditions, without producing any simulated human conclusions. The detailed [validation record](../../../artifacts/answer_repair/review_20261013/validation.json) is saved with the package.
