# First three real E teacher calls · 2026-10-27

The user supplied an existing Azure key through the local hidden terminal input. These are three real requests from the existing 500-ID Train pilot, not invented fixtures, additional test data, or SFT-approved targets. No key or gold reference is included here or was sent to the teacher.

All three returned exactly `gpt-6.1-sol-2026-09-29`, completed normally, and passed the frozen v2a format/source checks. The two supported verdicts and one insufficient verdict all kept the original answer. Therefore this connection check demonstrates no repair gain and does not test replacement behavior.

| Frozen ordinal | Verification | Final action | Visible target + Qwen EOS | Conservative accounting |
|---|---|---|---:|---:|
| 1 | insufficient | keep | 86 tokens | $0.014895 |
| 2 | supported | keep | 66 tokens | $0.011480 |
| 3 | supported | keep | 70 tokens | $0.014665 |

Total **$0.04104** is the client envelope at registered $5/M input and $15/M completion tokens, including reasoning once; it is not the Azure invoice. Actual provider usage and cache-write fields are preserved for later cost reconciliation. Targets are under 384 tokens using the pinned local Qwen tokenizer, without truncation.

- [Independent replay receipt](connection_audit.json) binds the exact execution registration, raw hashes, actions, budget and tokenizer revision.
- [Three original tasks](connection_tasks.jsonl), [raw provider responses](responses/), and [the original journal prefix](journal_prefix.jsonl) reproduce the checks without credentials or network. The prefix is a snapshot; the live journal may later have more requests.
- [File hashes](checksums.json), and [recorded-response regression](../../../tests/test_e_teacher_connection_recorded.py).

**Semantic approval is zero.** In particular, the Texans answer uses evidence about the historical Texans/Chiefs franchise. Its entity alignment conflicts with the offline reference's interpretation; it stays pending semantic review. The singers-versus-band wording also needs the frozen semantic policy rather than automatic gold-string filtering. The cathedral example illustrates that a gold-matching original can still lack sufficient support in the supplied evidence. Do not copy gold into a teacher request, or equate reference agreement with an evidence-verification label.

Next: continue from request 4 in the same journal to at most 500 requests / 15 USD conservative accounting; apply the previously frozen structural, semantic-review and length gates before expansion or SFT. The original client still stops on a failed request; no automatic retry, new ledger or prompt change is authorized by a passing connection check. Current production E remains unavailable and no adapter has been trained.
