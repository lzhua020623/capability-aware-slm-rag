# E teacher token-window transport · 2026-10-27

The current teacher runs on Azure. CPU/GPU selection on the sending computer is not the bottleneck; the legacy client deliberately sends at most once per 61 seconds. This optional transport changes pacing and terminal progress only, not teacher requests, cost accounting, target acceptance or training eligibility.

- [Profile](../../../configs/distillation/e_teacher_transport_v2.json) pins the unchanged parent registration. Serial single-flight requests reserve the entire input byte envelope plus max completion in a rolling 61-second / 95K-token window, at least five seconds between starts. Completed usage does not free unused output reservations early. The registered deployment is 100K TPM / 100 RPM; temporary Azure rate reductions can still stop the job.
- Same cumulative 500-ID / 15 USD journal and process lock; no new sampling, replay of completed requests or automatic retries. A fresh process cools down for 61 seconds before resuming. Profile and source hashes are journaled before sending.
- Resume/wait/completion messages contain counts, verdict/action and conservative cost only. No question, evidence, credential or hidden reasoning transcript is printed.
- Opt in using `--transport-profile token_window_v2`. The running legacy Python process is not hot-updated. It must finish its in-flight request before a restart, and the user must supply the existing key again because it is not stored. The old process was still running and was not interrupted by preparing this change.

[Simulation](pacing_simulation.json) covers the 490 tasks remaining at its snapshot: assuming 5/15/30 seconds service latency, about 1.71/2.06/4.10 hours versus about 8.30 hours for legacy pacing. This is an offline scheduler estimate, not measured Azure runtime or a GPU benchmark. Teacher schema success and throughput are not semantic accuracy or SFT approval.

**1,339 full tests passed in 135.815 seconds**, including 34 focused teacher checks; see [full log](full_tests.log), [focused log](targeted_tests.log), [tested sources](source_snapshot.json) and [post-work checks](post_selfcheck.json). The latter independently compares the AST of the original request builder, role preflight, cost and acceptance checks to `ada03c2`. Existing data, E contract, model/version, 4096-token teacher cap, 384-token student cap, quality gates and production configuration remain unchanged.

This transport follows [Microsoft's quota documentation](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/quota): rate estimation includes prompt and maximum output, not merely billed generated tokens. GPU-based Qwen inference/QLoRA remains a separate subsequent stage. No actual v2 API-throughput result is claimed by this archive.
