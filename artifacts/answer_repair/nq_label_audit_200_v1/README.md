# NQ initial-label stratified review 200

On 2026-10-22, 50 items were drawn from each of the four initial-label classes in the established development split, seed=42, and hash-shuffled separately after sampling. Only the ID, frozen usage and initial-label stratum were used; questions were not picked by EM, gold, answers or expected E gain. The original 100 questions, the 125 already-used challenge items and the separate preview questions were excluded, 226 known exposed IDs in total, and normalized question overlap was checked. Existing human records, the 25 deferred items and all old experiments are kept as is.

**Current: sampled 200/200; semantic assistance 30/200; human confirmation 0/200.** The remaining 170 items have had no semantic assistance, so the whole batch cannot be called fully reviewed. A1–A200 are new numbers for this batch, separate from the old N, F, H numbers.

- [Review page with assisted opinions for the first 30 items](A200_assisted_through_batch06.html): recommended page; human fields are left empty and assisted opinions are in the collapsed section. The [first-20 snapshot](A200_assisted_through_batch04.html) and [first-10 snapshot](A200_assisted_through_batch02.html) are retained. Each snapshot's review packet hash differs; existing drafts should first be exported on the original page and cannot be mixed directly across snapshots.
- [Original pending review page](A200_review.html): contains no assisted judgments.
- [Fixed sample list and exclusion scope](manifest.json), [original review packet](review_packet.json), [full Top-5 queue](review_queue.jsonl).
- `runtime.jsonl` contains only the five runtime fields; no initial labels, human opinions, gold or EM. E was not called on these 200 items in this round.
- [Source validation for the first 30 assisted items](A200_assisted_through_batch06.manifest.json) and [assisted distribution / initial-label comparison](assisted_summary_through_batch06.json). All excerpts passed passage ID and character-span checks; this only proves the citations come from the source text.

The review separately records original-answer correctness, whether Top-5 supports the original answer, whether Top-5 is sufficient, accessibility, the main label, the support location and reference issues. Correct but not supported by the current Top-5 is also an expressible combination; "has a citation" is not used in place of correctness. Initial labels are hidden by default in the source collapsed section; gold is collapsed separately. Browser drafts are saved on the local machine, and a batch is exported as JSON when finished; exports with a different review packet hash are refused from being mixed in.

The assistance for the first 30 items consists of Codex opinions with the reference visible, not independent blind review or human gold standard. It suggested 16 correct, 3 utilisation, 3 insufficiency, and 8 not yet classified; having a preliminary class does not mean moderate-confidence / scope disputes are resolved. Compared with the initial labels, of the 22 items with a clear assisted class, 7 agree and 15 differ; this cannot be treated as human review accuracy or extrapolated to the full set.

First 10: A1/A2/A4 lean correct; A3 role vs actor ambiguity; A5 leans toward time-range E error; A6 song title inconsistent with the retrieved material; A7 definition-question vs numeric-question ambiguity; A8 singer / credited artist boundary; A9 location granularity; A10 modern/historical dividing line and accessibility boundary. Newly added A11/A13/A16/A18/A20 lean correct; A12 lacks the year/league criterion; A14/A15 lack evidence for the asked entity/relation; A17 answered the tick as a disease; A19 is a grounded abstention, but did not give the person the QA asks for.

For A11 the reference gives a biological category, while C1 answered ATP/NADPH; for A16 the reference specifically means the seven gifts, while the original question has no seven qualifier. Both have the reference doubt recorded separately, without modifying the original gold or frozen EM. For A19, support=true means the abstention has an evidence basis, `response_kind=evidence_grounded_abstention`; it does not mean a supported person answer has been provided. Truthfulness/grounding is separate from task answer correctness, and this case cannot be treated as fabrication or E misuse.

Newly added A21/A22/A24/A26/A27/A29 lean correct; A21 still has a question-wording issue of "post-war" vs wartime plan; A22 is an alias for the frenulum of the tongue, A29 is another valid definition completion, and neither is judged wrong because the reference string differs. A23 population lacks an as-of date; A25 does not specify which nun; A28 does not define what kind of sound-picture combination "first" means; A30's not specified may be an in-material abstention, or may intend to say the author is unknown, and cannot be directly equated with anonymous. These four items have their adjudication withheld; the support=true of A25/A30 still does not mean the question ambiguity has been removed or that they are answered correctly.

The 50 per class across four classes is a stratified quota for quality audit, and does not represent the natural error distribution. The unweighted results of the 200 items must not be used directly to estimate the class proportions or overall accuracy of the 74k items. This batch is from now on recorded as exposed to E development, and cannot later be called a new unseen test. The original development/calibration/held-out split was not changed; team training exclusion and page and semantic near-duplicate isolation are still pending coordination.

Reproduction:

```sh
python scripts/prepare_nq_label_audit.py
python scripts/build_nq_label_audit_assisted.py --assisted data/annotations/assisted/nq_label_audit_200_v1_through_batch06_20261022.jsonl --out-stem A200_assisted_through_batch06
```

Inputs are the local original 15 shards, the verified initial-label layer and the established split. Outputs are byte-identical on rerun, and overwriting existing different content is refused. The original 200 items, subsequent assistance and human confirmation are saved separately.
