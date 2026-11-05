# Timeout of the 55th New Request · 2026-10-28

After the original 2,000 batch added 54 new complete responses, the 55th request hit a `TimeoutError`, with no response file. The original socket timeout was configured at 180 seconds; it cannot be confirmed whether the provider completed or billed it. A keyless GET to the current endpoint returned 401, which only shows it was reachable at check time.

Original 500 + 54 gives 554 complete candidates in total; the 54 have independently verified version, raw bytes, usage, contract, and cost. The known cumulative unit-price estimate is US$3.784529, plus the maximum reservation of US$0.13211 for the unknown request, still counted as US$3.916639; no refund, no resend, no re-collection.

- `original_attempts55.tar.gz`: the original journal and 54 responses, 55 whitelisted members, no key or launcher.
- `archive_manifest.json`: SHA-256 of the archive and of each member.
- `timeout_audit.json`: integrity, cost, and diagnostic limitations.

Recovery is not a bypass of the failure protection: this time one frozen failure was independently audited, all costs were kept, and that ID was explicitly marked as unresponsive and not eligible for training. The original ledger stays byte-identical, and later writes go to a new directory; new unknown failures still stop. The user subsequently authorized [cumulative 5,000/$40](../e_teacher_expansion_5000_v1/README.md), and recovery starts from new ordinal56.
