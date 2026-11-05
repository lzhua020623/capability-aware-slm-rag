# E output distillation preparation · 2026-10-27

Branch `controller_2`. This round starts the new NQ distillation extension: data, contracts and offline teacher tasks have been prepared, and the teacher deployment and a controlled pilot-production entry point have been set up on Azure; it does not overwrite the old r2 failure results, nor change the current E unavailable, C1 weights, retrieval resources or routing. **The 500-item real teacher pilot production is complete and independently replayed**, the 100-item semantic review is prepared, and the human gate, SFT and GPU training acceptance are not yet complete; see [2026-10-28 results](E_TEACHER_PILOT_RESULTS_20261028.md). The [first 3-item connection check](../artifacts/controller/e_teacher_sol_connection_v1/README.md) is kept as a historical snapshot.

## Component review conclusions

This round refreshed all remote branches of the main repository and checked two known forks: no new commits. Diagnosis `138d52a`, retrieval enhancement `37e469e`, reranking `5583285` and E `5da1fe5` are all in the current history; the integration layer keeps later compatibility fixes and is not required to be byte-identical to the original branch files.

The paragraph above records the first-round check. After the preparation commit, another refresh found three new reranking commits up to `1d51856`, for which [follow-up compatibility integration](RERANK_1D51856_INTEGRATION.md) was done: adding usage boundaries for the new selector evaluation, deep candidate pool resource registration and frozen backend parameter checks. The current E contract and samples are unchanged.

Merging the latest source does not mean all components can run in production: P60 lacks the original 78-dim online features and the used IDs of the three HGBs, and the startup check exits 2; the new reranking weights and answer selector are not promoted; E r2 did not pass real acceptance. The team baseline branch was not merged as a whole; the original NQ split was integrated independently by fingerprint, and its new context_fallback per-question outputs are still to be delivered.

Therefore E training data can be prepared independently, but a new end-to-end effect evaluation cannot be started. The component audit and run receipts are in [this round's archive](../artifacts/controller/e_distillation_v1/README.md).

## What is fixed this round

- The [preparation protocol](../configs/distillation/e_prep_v1.json) binds the source hashes of the existing v2a prompts, parser and data boundaries.
- The student is the existing `Qwen/Qwen2.5-7B-Instruct`, keeping the revision registered in the repository. The candidate method is QLoRA/SFT, training only E's adapter; C1 uses the original weights. The training library version, VRAM, adapter parameters and long-input scheme must be registered and measured before actual training.
- Inputs are still only sample_id, dataset, query, initial_answer and Top-5. The teacher has no gold, AI labels, Diagnosis results, tools or external retrieval.
- Model output keeps the five fields original_verification, verdict_evidence_ids, replacement_answer, replacement_evidence_ids and reason. supported is kept; contradicted is replaced only when there is a qualified candidate; insufficient is kept and flagged. The final action is executed by the program; no separate, contradictory action field is trained.
- The teacher gives a short verifiable reason of at most 60 words. The student output budget is still 384 tokens; the word limit cannot substitute for a real tokenizer check. The teacher's visible output and internal reasoning total budget are registered separately, and 384 is not taken by default as the teacher's total budget.
- The teacher chosen is Azure GPT-6.1 Sol `2026-09-29`, Global Standard pay-as-you-go, 100K TPM. The status of deployment `elec5623-e-teacher-sol-v1` has been verified as Succeeded; the original ELEC5623 project region does not support this model, so an existing compatible project under the user's same subscription is used, with no new resource created. Keys are not committed to the repository.

## Teacher call registration

The independent [execution registration](../configs/distillation/e_teacher_sol_pilot_v1.json) references the original preparation protocol and the full task fingerprints; the original preparation protocol stays byte-unchanged. Sol was chosen because it provides structured outputs and reasoning capability, and Azure's currently published standard short-input price is $2 per million input and $10 output, lower than Astra's $10/$50; this does not prove it is good enough for this task, and pilot-production acceptance is still required. [Microsoft pricing table](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/provisioned-throughput-sizing)

- **Version restriction**: Azure's current policy is still `OnceNewDefaultVersionAvailable`, and automatic upgrades could not be disabled on the page. The client requires every response.model to be exactly `gpt-6.1-sol-2026-09-29`; on mismatch it stops and that item is not accepted. The deployment itself cannot be described as forbidding automatic upgrades, nor can it be claimed that the version was verified in real time via the management API before the calls.
- **Budget**: this pilot-production ledger allows at most 500 requests and a client-side cost cap of 15 USD. Each request first reserves a token upper bound of the full input package UTF-8 byte count plus 4096, and 4096 total output tokens; requests whose input upper bound exceeds 90K are rejected. Conservative accounting at $5 input/$15 output per million covers the currently published input cache-write and long-context prices. After success, settlement is by usage; reasoning is already in completion_tokens and is not double-counted. This is an upper-bound estimate under the currently registered prices; it cannot be treated as the Azure bill, and does not control spending by other applications or future price changes.
- **Failures**: timeouts, missing usage, version mismatch, refusal, truncation, illegal citations or contract non-compliance all stop immediately, keeping the full reserved cost; no automatic retries, and the original responses are not overwritten. When a process is interrupted, already-reserved requests are likewise blocked from being resent.
- **Default run**: without `--execute` it only validates. Actual calls default to a cumulative 3; continuing uses the same fixed directory and ledger and skips already-successful IDs; explicitly specifying a cumulative 500 expands to the full pilot production. The four frozen AI strata are called in rotation, so a budget stop does not leave only the first class. To stay below the deployment rate limit, the initial version uses a single process with at most 1 call per minute, so 500 calls take at least about 8.5 hours; this does not consume the local GPU.
- **Teacher and student budgets are separate**: the teacher's 4096 includes internal reasoning and visible output, with medium reasoning; only the visible JSON, usage and request information returned by the provider are saved, and no hidden chain of thought is requested. The student target must still be verified item by item to not exceed Qwen's 384 tokens.
- **Quality gate**: before expansion, the full provenance/contract compliance rate must be at least 95%; by fixed hash, review at most 50 replacements (at least 25 are required), and fill the remaining slots in rotation among three classes (supported and kept, insufficient evidence and kept, clearly wrong but no candidate) up to 100 in total; after the rare kept classes are fully included, the remainder is allocated. Each of the three verification labels has at least 10 items. At least 95 must be semantically correct, there must be no harmful repairs among the reviewed replacements, and pending counts as not passed; if there are fewer than 100 compliant outputs or repair/label coverage is insufficient, there is no expansion. Format checks do not approve SFT, and all training targets get a separate 384-token check. This gate is an engineering decision for this pilot production and cannot be used to claim an overall harmful-repair rate of zero.

```bash
# Local re-check only, no Azure calls
python scripts/run_e_teacher_pilot.py

# Configure the actual Azure v1 base endpoint in the local terminal; do not put the API key on the command line/in chat
export AZURE_OPENAI_ENDPOINT='https://<registered-resource>.services.ai.azure.com/openai/v1/'
python scripts/run_e_teacher_pilot.py --execute --prompt-key
# Subsequent full pilot production under the same ledger (the first three items are checked first; already-successful requests are not repeated)
python scripts/run_e_teacher_pilot.py --execute --prompt-key --total-requests 500
```

The endpoint must match the observed resource hash; other resources, redirects and URLs containing a key are rejected. `--prompt-key` reads the existing key via hidden terminal input and keeps it only in this process's memory; `AZURE_OPENAI_API_KEY` can also be configured in advance. The user has completed the first 3 real calls via hidden input; the key was not given to the assistant or written to the repository. Do not send the key in the conversation.

First check measurements: two supported and one insufficient, all keeping the original answer; the Qwen targets including EOS were 86, 66 and 70 tokens respectively; the conservative client-side cost totaled $0.04104, and the actual Azure bill is not verified. It only validates the connection and output contract, does not cover real replacements, and is not a teacher semantic pass. The Texans question cites the old Texans/Chiefs team chain, which carries an entity-alignment risk against the offline reference and must go into pending semantic review; it cannot be added to SFT just because the format passed. After that, continue from item 4 of the same ledger, with the full pilot-production quality gate unchanged.

## Throttling supplement · transport v2

The 500 items are remote Azure teacher calls; the local CPU only sends requests and does not run the teacher model. Switching to a Kaggle GPU would not remove the client-side waiting; the GPU is for later student inference and QLoRA. The original one item per 61 seconds was a conservative initial policy; after observing single-item service times of about 2.5–7.3 seconds, a new independent [transport v2 registration](../configs/distillation/e_teacher_transport_v2.json) was added, while the original teacher registration, request bodies, sample order, budget and quality gate all keep their original bytes.

v2 must be explicitly enabled: serial with at most one request in flight, the total full token reservations in any rolling 61 seconds not exceeding 95K, no more than 60 requests, and start times at least 5 seconds apart. The reservation is the full request UTF-8 byte upper bound + 4096 input allowance + 4096 maximum output; rate reservations are not refunded early based on small actual outputs. The estimated range for all 500 items is 15,705–22,922 tokens/item, all passing the window check. The [Microsoft documentation](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/quota) states that rate estimation includes the input and maximum output, which differs from billed usage.

```bash
python scripts/run_e_teacher_pilot.py --execute --prompt-key \
  --total-requests 500 --transport-profile token_window_v2
```

Continue using the same directory, process lock and cumulative cost ledger; completed samples are not resent. On resume, first wait one full window, clearing rate times unknown to the old process, then start sending. The new profile and the client/scheduler source hashes are appended to the ledger before sending. The original stop-on-failure, no automatic retries and no stored credentials remain unchanged. Azure temporary rate limiting can still trigger a stop; the local 95K estimate cannot be written as a service guarantee.

The terminal now prints resume, rate_wait and completed progress, containing only cumulative counts, labels/actions and conservative cost, without printing questions, evidence or credentials. A running old Python process will not hot-update; to switch, stop the old process when there are no unsettled requests, then start v2 with the same ledger and re-enter the key via hidden input. Do not start two writers in parallel, and do not force-kill unsettled requests and then automatically resend them.

Offline simulation of the remaining 490 items: with single-item service time set to 5/15/30 seconds, v2 takes about 1.71/2.06/4.10 hours respectively; the old policy about 8.30 hours. This is a scheduling simulation for the current input shape, not an end-to-end GPU or Azure measurement, and does not promise a fixed completion time. Real connection and teacher semantic results continue to be recorded per the original protocol.

## Sample arrangement

The team splits and old held-out pool restrictions are unchanged, using the [NQ usage boundaries](TEAM_DATA_BOUNDARIES.md).

| List | Count | Sampling and purpose |
|---|---:|---|
| Teacher pilot | 500 | From the allowed 46,421 Train, 125 for each of the four AI initial labels; only for teacher quality/cost pilot production |
| Development | 500 | Uniformly drawn by fixed hash from the allowed 9,440 Validation; development evaluation |
| Selection | 500 | Drawn from the same allowed Validation pool, not overlapping Development; one candidate confirmation |
| Final test | 11,114 | Kept sealed; current tools cannot read its questions/evidence/gold |

Sampling reads only IDs and AI metadata, and does not filter questions by gold, answers or trial-run scores. The pilot is an intentionally balanced diagnostic sample, not representative of the natural distribution, let alone 500 confirmed utilisation items. Development/selection run are still usage subsets of the team validation, and the team data is not re-split.

After passing the teacher pilot production, the candidate target is 5,000 qualified mixed supervision samples, capped at 10,000; the expansion list, budget and quality gate must be sealed separately. This round's 500-item pilot production cannot be treated as the full training scale, and AI labels cannot be copied directly as training targets.

## Tools and outputs

```bash
python scripts/prepare_e_distillation.py \
  --manifest artifacts/controller/e_distillation_v1/sample_manifest.json \
  --export-pilot data/processed/e_distillation_v1/pilot_20261027
```

When re-validating the list, pass only `--manifest`; the output directory must be brand new, and an existing pilot production cannot be overwritten.

- `teacher_tasks.jsonl`: five-field runtime, original student prompt, teacher messages, object-root output schema and content fingerprint. It is an offline task, not an API Batch file that can be submitted directly.
- `offline_references.jsonl`: physically separated train gold/AI metadata, only for offline screening, and must not be uploaded with the teacher tasks.
- `inputs_manifest.json`: hashes of the original 15 shards, the sample list fingerprint, hashes of the two exported files and to-dos. The full tasks/references are saved in the ignored data/processed; the repository only records IDs and audit summaries.

The tool first excludes Validation/Test/held-out pool by ID, then decodes Train. Before teacher prompt generation, offline fields are removed by an allowlist. The existing student v2a schema is kept unchanged; the teacher transport schema uses an object root, and cross-field and provenance constraints continue to be checked locally.

`audit_teacher_output()` only certifies format and provenance, and always returns semantic_verified=false, approved_for_SFT=false. It can detect nonexistent citations, snippets not in the original text, wrong changed-answer branches and equivalent changed answers; it cannot prove that the question relation, qualifiers or answer meaning are correct.

## Next steps and stopping boundaries

1. Teacher deployment, execution budget, provenance guards and API transport are registered and implemented; the first 3-item connection/contract check is complete and independently replayed; no silent model switching, format relaxation or opening of a separate ledger.
2. The 500 items under the same ledger are complete, with a ledger cost of $6.58152; the 100-item quality review has been generated (49 replacements, 21 supported kept, 21 insufficient kept, 9 no candidate). Current human confirmations are 0; the results cannot be used to modify this round's gate, and there is no automatic conversion to SFT.
3. Only after the teacher capability and provenance checks pass, expand and build the SFT data; the tokenizer checks all target lengths, and truncation must not remove the supporting basis while still keeping the original supervision.
4. Measure the training environment and freeze at most two training configs, engineering fixtures, the harmful-repair cap on correct questions, confidence intervals and promotion rules. All these fields must be made concrete before any training; this preparation protocol does not pose as the final execution seal.
5. Fix C1, Top-5, prompts, decoding and budget, and compare keeping C1, the original E and the distilled E. Development selects one candidate, Selection confirms once; also report semantic repairs/harmful repairs, EM/F1, coverage and cost.
6. After targets are met, integrate into the fixed routing; while P60 and the common baseline are incomplete, no integration gain is announced. FEVER does not enter this round until it is unified. DPO, RSI and knowledge-base modification are listed separately and are not changed at the same time.

The teacher API preparation is based on [OpenAI Docs Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs): transport format constraints do not prove semantic correctness, and refusals/truncations still need separate handling. The training implementation references [TRL SFT](https://huggingface.co/docs/trl/main/en/sft_trainer) and [PEFT quantization/QLoRA](https://huggingface.co/docs/peft/main/en/developer_guides/quantization); the actual environment must be version-locked before acceptance.
