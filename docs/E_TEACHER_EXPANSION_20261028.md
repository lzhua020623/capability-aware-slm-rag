# E expansion and timeout recovery · 2026-10-28

The latest user instruction is "**可以直接上5000**", overriding the earlier "先2000条 有效再扩量". The cumulative 5,000/US$40 option previously confirmed by the user is kept: the original 500-item pilot production is reused, with at most 4,500 subsequent requests (**including the 55 already attempted**); 5,000 is not exceeded automatically, and SFT is not approved automatically. Extrapolating from the original pilot usage, the cumulative total is about US$34.27; the actual bill is not verified, and unknown costs keep their full reservation. Registered in the [5000 config](../configs/distillation/e_expansion_5000_v1.json) and [this round's confirmation](../artifacts/controller/e_teacher_expansion_5000_v1/user_acceptance.json).

## Reason for this stop and what is kept

The 2,000-item collection hit a `TimeoutError` on the newly added 55th request, and no response was saved for it. The original client socket wait cap was 180 seconds; the record cannot distinguish connection timeout, read timeout, temporary network interruption or remote latency, and cannot be attributed to an Azure fault. A key-less GET check received 401 from the current endpoint, showing that the endpoint was reachable at check time, which does not prove the original request was not billed. The problem is not a budget overrun or insufficient GPU memory.

- Original 500 + 54 new, 554 complete candidates in total; the model version, raw response hashes, usage, provenance contract and settlement of the 54 new items have been independently replayed.
- All 54 new items passed structure/provenance: 38 supported, 6 contradicted, 10 insufficient; 49 kept, 5 replaced. Student gain was not measured on this basis, and semantic targets were not automatically approved.
- Verifiable cumulative unit-price estimate US$3.784529; the 55th request's full reservation of US$0.13211 continues to occupy the budget, so the cumulative total is counted as US$3.916639.
- The [raw archive and timeout audit](../artifacts/controller/e_teacher_expansion_timeout_20261028/README.md) saves the original journal and the 54 responses; the old ledger, old config and execution seal keep their original bytes.

## Recovery and scale boundaries

The recovery version only skips the 55th request, which has been audited individually: no resend, no refund, no added replacement questions; it continues from the subsequent 56th, **sending at most 4,445 more**. The new 4,500-ID list is a fixed extension that keeps the original 1,500-ID list as its prefix; all IDs come from the allowed Train, 1,125 per AI metadata class, and after reusing the original 500, each stratum's request quota totals 1,250. Sampling uses only IDs and frozen AI metadata, and does not filter questions by gold, predictions or EM. Development500, Selection500 and the team Test of 11,114 items are all excluded.

Because 1 response is already unavailable, even if all subsequent requests succeed, at most **4,999 complete candidates** are obtained; format failures or budget stops would reduce the count further, and the 5,000 call slots cannot be written as 5,000 qualified SFT items.

This change only affects transport execution: the subsequent request timeout increases from 180 seconds to 600 seconds; the model, version, prompts, original Top-5, reasoning, 4096 total completion budget, student 384-token budget and token-window v2 are all unchanged. Request start time, failure stage, duration, exception type and HTTP status when available are added; keys, error response bodies and arbitrary exception strings are not saved. A longer timeout cannot guarantee request success.

The original failure ledger is imported read-only; the new ledger uses an independent directory and append-only events; during execution, both the original directory lock and the new directory lock are held. Each startup replays the original 54 items and the newly completed responses, checking cost, identity, order, source and input fingerprints, without double billing. A newly occurring unknown-cost request still stops immediately and needs a separate audit; **the skip rule of the 55th request is not applied automatically**.

```bash
# By default only validate inputs, provenance and the old ledger; no Azure calls
python scripts/run_e_teacher_5000.py

# Actual collection: enter the existing key via hidden input in the local terminal
python scripts/run_e_teacher_5000.py --execute --prompt-key
```

The new client price-estimation base includes the original 500, the 54 completed and the full reservation for the unknown cost; before each send, the maximum input/output is still reserved at the conservative US$5/15 rate, and then settled at the fixed unit-price estimate. It stops upon reaching US$40. Serial with at most one request in flight, full reservations within a rolling 61 seconds not exceeding 95K, start intervals of at least 5 seconds, resume first waits one window, automatic retries 0.

## Cost and quality basis

The [original 500-item cost reconciliation](../artifacts/controller/e_teacher_expansion_release_20261028/cost_reconciliation.json) lists the old conservative ledger of US$6.58152 and the official unit-price estimate of US$3.42705 separately, without rewriting the old records. According to [Microsoft's published short-context prices](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/provisioned-throughput-sizing), standard input/cache read/cache write/output per million are US$2/0.10/2.50/10 respectively; cache writes are a subset of input and reasoning is a subset of completion, not double-billed. The actual Azure bill is not verified; if the price or service tier does not match the registration, it stops.

For the frozen 100 items, the user previously stated "已完成审核，按结果登记" and "算全部通过 我在网页里面没有标注"; per the [chat confirmation](../artifacts/controller/e_teacher_expansion_release_20261028/user_batch_release.json), candidate collection was released, without generating nonexistent per-item web judgments or a 100% measured semantic accuracy. The two items disputed by the assistant, T011 and Texans/Chiefs, remain under temporary training-entry isolation. New targets still need provenance, 384-token length, semantic and cross-set checks; pending items are listed separately and do not automatically enter SFT.

Only the Qwen E adapter is trained; C1 keeps the original weights. The training config and numeric promotion thresholds are registered before training; Development500 compares the repairs, harmful repairs, EM/F1 and cost of C1/original E/distilled E; Selection500 gives only the final candidate one confirmation. Test is not used for distillation or prompt modification. The user's permission this time to collect directly up to the 5,000 cap does not mean the student is already effective, nor does it authorize further expansion to more data.

The old 2,000 config, task package and budget records are in the [historical seal](../artifacts/controller/e_teacher_expansion_2000_v1/README.md) and are not overwritten; do not use the old launch script again to try to recover from this failure.
