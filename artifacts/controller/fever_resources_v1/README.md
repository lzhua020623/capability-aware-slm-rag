# FEVER Team Resource Acceptance

- `import_validation.json`: size/SHA-256 of the two original ZIPs, CRC/hash acceptance of the selected members, full order and text checks of the 5,396,106 corpus/ID entries, and the offsets fingerprint.
- `original_metadata.json`: the 601-byte index-build record taken from the original package, byte-identical to the repository's original `results/retrieval/fever_bge_ivfpq_meta.json`.
- `structure_validation.json`: actually loads IndexIVFPQ and checks count, dimension, IP, PQ parameters, nprobe, and by_residual; searches Top20 with three reconstructed PQ vectors and then reads back the original text.
- `import.log` / `structure.log`: the corresponding actual execution logs.

Local resources are at `data/processed/controller_resources/fever_v1/`; large resources and offsets do not go into the original Git. For reproduction commands, see the [integration notes](../../../docs/CONTROLLER_RESOURCE_ADAPTER_V1.md). The two original ZIPs are kept in the local Downloads, sourced from the team Drive; when migration is needed, reuse the original packages and verify them by hash.

The remaining in `import_validation.json` is the record at the time the import completed; the structural/search checks passed afterwards. The original FEVER BGE revision is still not recorded; this round does not claim to have done semantic BGE compatibility acceptance or QA quality evaluation. The old fever_dev in the ZIP only has its fingerprint recorded and does not replace the T2 frozen source. The FEVER provider, the new C1, and the formal execution seal are still needed later.
