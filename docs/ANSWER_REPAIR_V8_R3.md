# v8r3: candidate grammar and rejection classification

The 32 real outputs of v8r2 are complete. The entry check returned to normal, but of the 9 candidate rejections, 6 omitted field names and 3 did not satisfy the source constraint. r2's pre-set gate is still a failure, and the original results and code bundle keep their original bytes.

## This round only handles the engineering representation

When a candidate has exactly three lines, three forms are explicitly accepted: full RANK/QUOTE/ANSWER; a bare integer first line plus labelled QUOTE/ANSWER; three lines of bare values in the fixed order rank/quote/answer. A bare first line can still only be a single integer or an integer in one level of square brackets. Partial mixes, prose explanations, multiple repeated sets or conflicting outputs are neither guessed nor clipped.

All three forms enter the same source address, contiguous source fragment, answer containment, word-boundary and length checks; negation, entity, concatenation, rewriting or approximate matching are not relaxed. The generation stopper also recognizes these three explicit grammars, stopping only after observing the complete final-line newline during generation; the actual raw generation is kept.

New stage rejection types: `format` means the structure/fields cannot be parsed, `grounding` means the citation or answer does not satisfy the source constraint, `constraint` means the length requirement is exceeded. All three keep the existing fallback behaviour; source rejections are not disguised as successful format fixes and do not count as valid changed answers. Stage records still keep errors, raw generations and costs, supporting frozen replay.

The six semantic check prompts, call order, model, inputs and budget are not changed in this round. Problems such as wrong acceptance on work identity, wrong rejection on comparison relations, and replacing terms with descriptions still exist and must be studied separately; they cannot be credited to or blamed on format compatibility.

## New prospective engineering gates

Still uses the same 8 items, four groups with 32 records, at most 64 generations. r2 treated all validation rejections as engineering failures, mixing unparseable structures with normal source protection. r3 distinguishes the two in the new protocol: format errors and unclassified rejections must be zero; each stage type has at least one valid call, and there is at least one complete three-stage path; outputs are complete, raw generations are not clipped, stop observations are complete and the budget is not exceeded.

Source or length rejections can be normal conservative fallbacks, but must be reported by type and cannot be counted as repairs or hidden. This definition only applies to new r3 runs and does not recompute r2 as passing. An engineering pass only means a larger development measurement can be carried out; it does not show that the semantic policy is effective, or that it meets the conditions for promotion to default.

Preparation entry point: `python scripts/prepare_answer_repair_v8_r3.py`. The separate protocol is `configs/answer_repair_v8r3_smoke8_v1.json`, and the bundle, Notebook and results go into separate r3 paths. r2 and the original 75-item strict experiment remain unchanged. New final answers are still left for human review; the 25 old unresolved items do not all need to be finalized first.

## Real execution results

Kaggle version 1 ran successfully in 553.1 seconds. 32 real outputs, 43 generations; the frozen CPU replay and the new engineering gates passed; v8 format errors were 0, and the 3 strict source rejections were kept. The full pipeline only corrected N96, EM 1→1; work-identity and comparison-relation misjudgements still exist, see [full results](ANSWER_REPAIR_RESULTS_20261019_V8_R3.md). The 75-item rerun was not started automatically, and the default policy was not promoted.
