# Controller T2 Data Package v1

For usage instructions and limitations, see the [data freeze document](../../../docs/CONTROLLER_DATA_V1.md).

- NQ and FEVER each have 200 development, 200 selection-run, and 500 final-evaluation subsets.
- The full held-out pool is NQ 7,564 (10.21%) and FEVER 1,886 (14.26% of valid candidates). All IDs in `evaluation_reserve_ids.json` are prohibited from training/tuning; records not drawn remain held out.
- `nq/` is the actual C1 runtime input; `fever/` is the query-only task for the C1 yet to be generated.
- `offline/` holds independent reference answers/labels, which must not enter the Controller or model prompts.
- `final_candidate/` is used only for the final version and cannot be used for development debugging. The file name does not imply it may be used for other purposes.
- `manifest.json` / `policy.json` / `exposure_audit.json` / `exclusions.jsonl` store sources, sampling, exposure, and exclusion reasons.
- `asset_inventory.json` records missing/cloud-placeholder resources, the not-yet-accepted matching index, and the BGE revision.

The generator refuses to overwrite existing directories. The registration entry is [controller_data_v1.json](../../../configs/splits/controller_data_v1.json), which binds the manifest SHA-256.
