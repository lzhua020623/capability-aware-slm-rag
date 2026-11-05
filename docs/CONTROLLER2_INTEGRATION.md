# Controller 2 integration handover · 2026-10-27

**Current production Adaptive RAG status and entry point**: see [CONTROLLER_1024_BASELINE.md](CONTROLLER_1024_BASELINE.md) (E2E production path passed). The text below retains the historical handover record of the branch merge and compatibility handling.

Subsequent work happens on `controller_2`. This round branched from `controller@5da8fa690bb04bd65f19ee9629b899d569fa0abe` and merged in the latest content of the two enhancement routes; the merge notes below describe the compatibility integration at that time, and outdated statements in them such as "E unavailable / P60 not delivered" are not authoritative.

| Source | Frozen commit | Added this round |
|---|---|---|
| `feature/retrieval-enhancement` | `37e469e0bd101dc5a905f95aeca1cafdc6a07e9d` | 3 commits: generation replay, Windows compatibility, historical GPU outputs, scoring and routing-weighted audit |
| `feature/reranking-enhancement` | `55832850640f5fe37c52a02779cc389a2e5f0276` | 18 commits: reranking training, answer-level and harmful-repair evaluation, random-reranking/sentence-trimming controls, candidate traces, unified adapter, answer-selector training tools |

The retrieval-enhancement merge commit is `5c793c7`. The second merge commit contains both the compatibility handling, tests and this handover described below; the final commit ID is as recorded in Git history.

## Compatibility handling

- Retain the original P3-20 FEVER support, label parsing, failure fallback, call metering and lazy package imports. The latest three upstream commits mainly add experiment tools and evidence and do not override the Controller's existing cross-dataset adaptation.
- `RefinementConfig/Result/Status` take `refinement_schema.py` as the single type definition; `schema.py` and the old package-level imports continue to point to the same types. Old constructor calls for accepted results can determine candidate_answer from final_answer; for rejections with no recorded candidate it remains unknown, and rejected content is not fabricated.
- New traces retain rejected candidate answers; regeneration can also be triggered when evidence IDs are unchanged but the text has been trimmed. New capabilities are enabled by explicit experiment configuration.
- Retain `EnhancementResult`'s rejection of non-finite latency, and the existing executor's consistency checks on actions, answer changes and failure fallback.
- The standalone retrieval-enhancement run package adds the split-out `src/generation/prompts.py`. The Windows original-package audit still checks the approved source hashes; a newer checkout will not be passed off as the historical original package.
- The historical T5 wrapper regression restores source from the original frozen package and also checks the old hashes before filling in missing files; a new check confirms that the current Controller source cannot rebuild the old sealed package. Neither the production sealing check nor the historical packages were relaxed or overwritten.
- Retain the repository's `AGENTS.md`, old-import regressions and all historical experiment archives, and merge the rules from both sides of `.gitignore`; the user's uncommitted assisted-review files are not included in this round's commits.

## Current actual runtime configuration

`configs/controller/current_baseline.json` and `p60_online_adapter.json` keep their pre-merge bytes.

- The subsequent baseline is still 10/24 Diagnosis/P60/P64, Probe OFF.
- The original 78-dimensional online feature provider and the training/tuning/calibration/analysis IDs of the three HGBs are still not delivered; the production pre-check exits with 2; no fictitious features are called and there is no fallback to the old routing.
- E for NQ/FEVER is still unavailable. Distillation, LoRA, Azure teacher calls and new GPU runs have not been started.
- Default reranking still uses the registered `BAAI/bge-reranker-large`, keeps Top-5, and does not enable trimming. Newly trained reranking weights are not delivered/registered; the answer selector currently has a training script and data statistics, with no accepted model.
- Do not treat reranking ranking metrics, random controls or the retrieval-enhancement routing-weighted estimates as gains of the current integrated system. Source results are interpreted in their respective reports, and absolute scores across environments are not directly concatenated.

## Local verification and optional dependencies

Targeted checks and the full regression record are in the [verification archive](../artifacts/controller/controller2_integration_v1/README.md). These tests use fictitious questions/providers, historical frozen assets and local audits, with no teacher API calls or GPU inference.

The new random sentence-trimming control uses NLTK's English sentence tokenizer, which must be prepared explicitly beyond the existing project dependencies:

```bash
python -m pip install -r requirements-refinement-experiments.txt
python -m nltk.downloader punkt_tab
```

Model code does not download this resource automatically at runtime; for offline reproduction, prepare the resource in advance and configure `NLTK_DATA`. This round's test environment and resource verification are recorded in the archive. The original `requirements.txt` already declares pandas/openpyxl, and the full test environment also includes these dependencies.

## Follow-up tasks

1. Hook up the original NQ split from the team's `feature/baseline-evaluation@387ddaf`, fixing Train 51,866, Validation 11,117, Test 11,114; obtain the matched RAG predictions/metrics/run_config and read evidence back by passage ID. This third branch is not within the scope of this two-route merge.
2. Register historical exposure and cross-branch training IDs. 69 old E registrations, 34 Controller development items and 27 selection-run items fall into the team test, at least 130 in total across the three; the full test cannot be called completely unseen by E. The old held-out pool remains protected; FEVER will be unified separately.
3. Check the data-generation/training IDs of the new reranker and answer selector against the team split; the local splits in existing scripts cannot replace the team's 70/15/15. Candidates are registered only after weights, near-duplicate isolation, data provenance and budget pass verification.
4. In parallel, complete the two P60 materials, and prepare E's short-output contract and teacher qualification validation. E training uses only permitted Train data, with Validation used for the selection run; Test does not enter training or prompt design.
5. Validate E candidates under a fixed module/routing configuration, complete engineering and real-effect acceptance, and then freeze the new version and the final team paired evaluation protocol.

The old T1–T5 packages, scores and stopping conclusions are kept as is. The old final candidates of 500 each belong to the original protocol; the scope, comparison groups, GPU budget and statistical method for the team's new test need to be sealed separately. For main-line progress see [T1–T9](CONTROLLER_TASKS.md).
