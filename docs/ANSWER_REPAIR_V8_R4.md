# v8r4: final round FACT verification prompt experiment

Real trial run completed: 32 records passed the engineering checks and frozen replay; the N31/N48 regressions failed, and only N12 passed. Per the pre-set rules, r3 is finally adopted; the final 75 items have been submitted separately. See [results report](ANSWER_REPAIR_RESULTS_20261020_V8_R4.md). The original plan and the infrastructure retry notes are kept below.

This version only changes the verification prompt and its ten-line parsing/stopping contract. It first writes FACT (the fact that the evidence establishes for the question), then compares the answer currently under review and gives the six judgements. Candidate verification passes in the candidate answer; no model calls are added, no new independent verifier is added, and the program does not treat FACT as verified truth.

The shared SEMANTICS, the minimal repair prompt, the three candidate grammars, source localization, decision rules, model revision, greedy/seed, the 384-token limit, the three stages and the four-group protocol all stay as in r3. The frozen build checks item by item the AST of unchanged functions, the constants, and the hashes of the 8 minimal repair prompts.

Fixed 8 development samples N1/N2/N8/N12/N15/N31/N48/N96, four groups with 32 outputs, at most 64 generations. Carries over r3's engineering gates; the full pipeline additionally checks only three pre-determined stage behaviours: N31 original check unsupported; N48 original check unsupported and Govind Ballabh Pant Sagar candidate supported; N12 original check supported and original answer kept. The other five and all ablations are fully recorded, and EM/F1 do not participate in version selection. A subsequent source rejection of N31's candidate is still honestly counted as a repair failure and is not conflated with an original-check regression failure.

This is the last time these 8 samples are used for prompt selection. r4 is selected only if engineering and all three regressions pass, otherwise r3 is kept, and adding r5 on this batch is forbidden. Infrastructure faults allow resumption with the same fingerprint and same configuration, but changing the policy or samples based on scores is not allowed.

The final version then runs v7, full v8 and the two ablations on the fixed 75 development inputs, with C1 reusing existing records; modifications stop after the run. The unseen stratified challenge set, the changed-answer union review and the team end-to-end comparison are frozen separately; smoke, development numbers or reference-visible assisted review cannot be written up as independent gains.

Preparation entry point: `scripts/prepare_answer_repair_v8_r4.py`. Protocol: `configs/answer_repair_v8r4_smoke8_v1.json`. The original r3 and earlier bundles, predictions, gold and scoring rules all keep their original bytes.
## 2026-10-20 infrastructure retry notes

Kaggle version 1 `360053773` failed after about 41.4 seconds of execution at the AST fingerprint check among the 118 portable checks; the CUDA environment check passed, and the number of model generations was 0. The Python 3.13 AST dump representation differs from that of Python 3.11 used when generating the fingerprint; the same fingerprint difference was reproduced exactly on local 3.13. This failure has no semantic results, and r3/r4 cannot be selected based on it.

Version 2 `360064691` uses the same `d13880c5…` frozen bundle. The new adapter serializes the AST under 3.11 rules only in a separate unittest process: it keeps the existing structure checks, omits the newly added empty type_params, and rejects non-empty parameters. No file in the frozen bundle was modified, and the separately launched model process is unaffected. The original 118 frozen checks all passed locally on Python 3.11, 3.12 and 3.13, plus two adapter guard checks.

The retry Notebook, adapter fingerprint and first-round failure log are saved separately. The retry finished in 577.3 seconds, and the result selected r3 according to the original three regressions and engineering gates, with no additional r5 and without looking at EM to select the version. The final 75 items have been submitted as `360070600`, using the same test compatibility approach, with all 29 inference source files matching the selected r3 one by one.
