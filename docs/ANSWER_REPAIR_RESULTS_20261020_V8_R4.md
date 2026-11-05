# r4 results and final version selection: adopt frozen r3

On 2026-10-20, the final round r4 completed 32 real outputs across four groups on the fixed 8 development samples, with 38 generations in total. All records passed the frozen-code CPU replay and the engineering gates passed; of the pre-declared N31/N48/N12 regressions, only N12 passed. **Per the pre-run rules, the final component experiment adopts r3, with no additional r5.** This decision was made before looking at this round's EM; these 8 already-seen development samples cannot be treated as an independent effectiveness evaluation.

## Three pre-declared regressions

| Sample | Expected behaviour | Actual r4 behaviour | Acceptance |
|---|---|---|---|
| N31 | Original answer unsupported | Still supported; FACT written as Charlie Rich's "The Most Beautiful Girl" | Failed |
| N48 | Original answer unsupported, correct candidate supported | Original answer rejected, but the Govind Ballabh Pant Sagar candidate is still judged unsupported | Failed |
| N12 | Original answer supported, keep the original answer | supported, kept Holloway Sanatorium | Passed |

N31's question is "who sings the most beautiful girl in the world". FACT did not keep the song title being asked about and still generated a summary around a similar song title. N48's candidate verification explicitly set Govind Ballabh Pant Sagar as the answer under review, but FACT still revolved around Dhebar Lake's second-largest description and wrote a judgement with no evidence of a larger lake. It did not fully use the same Top-5. FACT itself can be wrong, and the same model's subsequent six judgements remain affected by this erroneous fact summary; adding FACT did not resolve these two known semantic problems.

## Execution, freezing and replay

The first Kaggle version `360053773` stopped after about 41.4 seconds of actual execution because the Python 3.13 AST dump differs from the local 3.11 fingerprint representation; the number of model generations was 0. The failure was reproduced exactly on local 3.13; it is a pre-check compatibility fault, not a semantic effectiveness failure.

The retry version [360064691](https://www.kaggle.com/code/yearo111/e-v8r4-smoke8-fact-final-20261019/log?scriptVersionId=360064691) ran successfully in 577.3 seconds, T4 ×2. The old AST representation rule is used only in a separate unittest process, and none of the original 118 checks were removed; the bytes of the frozen bundle, model process, inputs, prompts and budget are all unchanged. This handling passed the original 118 checks locally on 3.11/3.12/3.13.

- Original frozen bundle SHA-256: `d13880c5e114b5f8379f931103af249e1f10b4975de98c9a5b74a9c161900e11`.
- Successful result ZIP SHA-256: `135f5dfb36aa89f0f8e1261188cafdd9adefe316c0ad41ea1d842dcd40778982`.
- Same Qwen/Qwen2.5-7B-Instruct, revision `a09a35458c702b33eeacc393d103063234e8bc28`, NF4/greedy/seed 42/384 tokens; actual C1 Top-5, no gold/human labels at runtime.
- The 32 records pass the original frozen replay policy; the acceptance gates recompute identically and the changed-answer queue is verbatim identical. One F1 aggregate in the derived summary differs in the last floating-point digit across Python versions, within the original frozen replay tolerance. The raw export files were not rewritten.
- Valid v8 stages: 12 original checks, 7 candidate generations, 4 candidate verifications; 7 source-localization rejections, of which 4 were original checks and 3 candidate generations. 0 format rejections, 1 complete three-stage path, 0 post-hoc truncations.

## Development description after version selection

The table below describes only the 8 already-seen samples; EM is strict reference matching, and a changed answer is not a human-confirmed repair.

| Group | EM correct | Changed answers | Generation calls |
|---|---:|---:|---:|
| v7 | 1→2 | 5 | 8 |
| Full r4 | 1→1 | 0 | 10 |
| No original check | 1→0 | 2 | 11 |
| No candidate check | 1→2 | 1 | 9 |

In full r4, 5 records were judged as original answer already supported, 2 original-check citations failed contiguous source localization, and 1 candidate was rejected; in the end there were no changed answers. The original-check citations for N2 and N8 were not valid verbatim fragments. The 8 cross-group changed-answer records remain review material only; this round adds no new human confirmation or conclusion of a reliable net gain.

## Next steps and handoff

The final 75 samples use the r3 inference source code. C1 reuses existing predictions; v7, full r3 and the two ablations each generate 75, for 300 outputs in total and at most 600 generations. The 29 inference source files of frozen bundle `b5be903d…` were checked one by one as identical to the selected r3, and passed the selected version's original 111 portable checks under Python 3.13. The batch has been submitted as [Kaggle 360070600](https://www.kaggle.com/code/yearo111/e-final-nq75-r3-comparison-20261020/log?scriptVersionId=360070600); results have not yet been collected, and the final development numbers must not be filled in early.

After the development batch the algorithm will not be changed any further; per the final protocol, run an E-internal unseen stratified challenge once, and review the full union of changed answers from v7/final v8. Final experiment version selection is separate from default-policy promotion; there is no conclusion yet on default-policy promotion.

Raw results, failure logs, replays and acceptance evidence are in `artifacts/answer_repair/v8r4_smoke8/`; the final development frozen bundle and run status are in `artifacts/answer_repair/final_nq75/`.
