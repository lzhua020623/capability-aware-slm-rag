# NQ full location mapping and provider acceptance

`validation.json`: checks the IDs, true UTF-8 line boundaries and body text of all 1,224,119 passages, and actually performs 1,224,119 text-mode seek read-backs; verifies corpus/index/ID/offset SHA-256. Not all vectors were re-encoded; the original 64-passage independent vector comparison information is retained in the report.

`preflight/manifest.json`: preflight of the new explicit NQ Train resource registration entry point on all 200 development inputs; Qwen was not loaded and no answers were generated. The source hashes record the working tree at acceptance time; the commit field is the parent commit before the change, and it cannot be used alone to reproduce the code added this time.

`retrieval_replay.json`: real BGE → original FlatIP → lazy offset provider replay of the first 10 questions of the development set; **for 10/10 questions the Top5 IDs and their order are identical to the old C1**, and the first five items of Top20 are also identical; no QA metrics. `retrieval_replay.log` keeps the actual run log.

For the full source registration, environment and regression results see [resource_adapter_v1](../resource_adapter_v1/README.md). The original NQ resources and recovery files are still in the Git-ignored `data/processed/`; the resource fingerprints and relative paths are in the [registry](../../../configs/controller/nq_train_resources_v1.json).
