# Azure E teacher registration · 2026-10-27

This archive covers teacher deployment/configuration and the tested pilot client, before the first live connection check. It is not a teacher-quality result or a trained adapter.

- [Deployment receipt](deployment_receipt.json): GPT-6.1 Sol 2026-09-29, `elec5623-e-teacher-sol-v1`, GlobalStandard, Succeeded, 100K TPM / 100 RPM. An existing compatible project was used; no new resource/subscription was created. Auto-upgrade remains `OnceNewDefaultVersionAvailable`; do not describe it as disabled.
- [Execution registration](../../../configs/distillation/e_teacher_sol_pilot_v1.json): one cumulative journal, up to 500 requests / 15 USD conservative accounting envelope; default three connection requests. Exact response model required, no automatic retries, no redirect or credential logging. Unknown charges retain the entire reservation and stop. This ledger is not an Azure invoice or an account-wide spending limit.
- [Offline preflight](offline_preflight.json): all 500 train-only tasks reconstructed against frozen runtime/prompt/schema fingerprints. No offline references or final test content sent. The original preparation and sample manifest remain unchanged.
- [Full regression](full_tests.log): **1,324 tests passed in 217.263 seconds** against `ada03c23f3fc16b716ff2c21888b6eb9432f4042`, including 19 teacher tests. Network calls are mocked; these tests do not prove Azure API compatibility or teacher semantics.
- [Source fingerprints](source_snapshot.json), [post-work checks](post_selfcheck.json), [archive checksums](checksums.json).

The computer-use tool rejected Terminal access, so the user must supply the existing API key in the local hidden-input launcher. No key was read/copied into the conversation or saved. Actual responses, when generated, stay in ignored `data/processed/e_distillation_v1/teacher_sol_pilot_20261027/`; separate audit is required before expansion or SFT. The initial launcher calls at most three requests, not the entire pilot. The current conservative one-request-per-minute client needs at least about 8.5 hours for 500 requests.

The current Controller E mode remains unavailable. P60 feature/provenance, new model-asset provenance, and the shared new baseline outputs still block production promotion. See [workflow and stop rules](../../../docs/E_DISTILLATION_V1.md).
