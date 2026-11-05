# NQ Train integration into E

**2026-10-22 update:** The 15 XLSX initial labels covering all 74,097 records have been integrated from `NQ_15批` on Drive; the original question, C1, Top-5, reference and old delivery match record by record. The initial labels contain 4,531 utilisation records, but they are mainly heuristically generated and are not a full manual gold standard; they disagree with the existing 75 human-review records on 36 records, and the human conclusions keep priority. The new data status, distribution and usage boundaries are in the [initial label integration report](ANSWER_REPAIR_NQ_PRELIMINARY_20261022.md). The "empty-label delivery" described below refers to the original source of October 17 and does not mean the latest Drive has no labels.

On 2026-10-17 all 15 shards from Google Drive were integrated, 74,097 NQ Train C1 records in total. The full set has been converted into E-compatible input, and the first batch of 100 development review materials has been fixed. The original materials keep an unconfirmed snapshot. As of the sixth batch on 2026-10-19, all 100 records have one round of human opinion: 75 finalized (11 internal E utilisation targets, 64 correct controls), 25 pending, 0 not yet viewed. The 25 include the original 17 deferred, 5 original criteria disagreements and the condition/supplement requirements of N58/N61/N83; the full data has not thereby become fully human-labelled. The 75 have completed the [v8 four-group real experiment and replay](ANSWER_REPAIR_RESULTS_20261019_V8.md); full v8 fell back as invalid on the original answer check in every case; the original 15 new changed answers await human review. The later [r3 real trial run](ANSWER_REPAIR_RESULTS_20261019_V8_R3.md) has passed the engineering threshold, and the full pipeline only corrected N96; the next step optimises semantic checking, and the 8 development debugging samples cannot be treated as final gain evidence.

The assistant's initial suggestion was 69 acceptable, 8 utilisation candidates and 23 deferred. The opinions of the first six batches are each saved separately, and the current status follows the cumulative review record; condition, completeness and expression disagreements and the assistant's semantic judgements are kept separately and not automatically included in the E core. N76 was checked against the user-provided same-entity source explanation, and the basis of interpretation is marked. The 75 finalized records can be used for development first, and the 25 remain isolated. See [review results, confirmation records and next-round constraints](ANSWER_REPAIR_NQ_TRAIN_REVIEW_20261017.md). The original data and gold on main stay unchanged as the source baseline; local review and adaptation are derived products.

## Source and checks

Upstream code and the delivery manifest come from `main@058f10baa47083220f95edb39db8f106aa7e85d6` and have been merged into the E branch. The original shards come from [nq_train_annotation on Google Drive](https://drive.google.com/drive/u/0/folders/1GRWOjia-PXfYh7YfvC1f3JzdjJqtJNDU). The SHA-256 of all 15 files matches the [delivery manifest](../data/annotations/nq_train_delivery/manifest.json), and the record-by-record field check found no anomalies; see the [full audit](../data/annotations/nq_train_delivery/drive_audit_20261017.json).

| Item | Count or result |
|---|---:|
| controlled samples selected from official NQ Train | 74,097 |
| upstream retrieval corpus | 1,224,119 passages |
| C1 generation failures / empty answers | 0 / 0 |
| reference passage in Top-5 | 46,633 (62.94%) |
| frozen EM match | 28,501 (38.46%) |
| EM mismatch and reference passage in Top-5 | 23,745 |
| upstream non-empty human diagnosis labels | 0 |

23,745 is the number of screening candidates and cannot be treated as the number of utilisation labels. EM=0 does not equal a semantic error, and hitting the reference passage does not mean the evidence is already clear and sufficient. The upstream 9,032 duplicate groups refer to identical **answer strings**; after case and whitespace normalisation the questions have no duplicates. There is no overlap in IDs or in these normalised questions with the old NQ3216, the 500-record pre-experiment, or the old E review/v7 inputs. Semantic near-duplicates and Wikipedia page grouping have not yet been checked.

The Train retrieval corpus is larger than the old dev corpus of 99,922 passages, so metrics should be reported separately from the old batches. The delivery records Qwen2.5-7B-Instruct, NF4, greedy, 64 tokens, but the model revision and full environment were not exported; the adapted results explicitly record the source C1 revision as unknown.

## E internal partition

First group by normalised question and partition by hash, then compute EM and screen review candidates. The hash does not read answers, gold or labels, seed=42, and 80/10/10 is the hash-bucket ratio; the actual counts are as follows.

| Use | Count |
|---|---:|
| development: development and rule selection | 59,188 |
| calibration: calibration after rules are fixed | 7,345 |
| reserved_evaluation: reserved evaluation | 7,564 |
| excluded_by_team: IDs separately reserved by the team | 0 |

The configuration is in the [frozen manifest](../configs/splits/answer_repair_nq_train_dev_v1.json), and the full ID table is saved locally in `data/processed/answer_repair/nq_train_v1/partition_ids.json`. The previously previewed `-1000144997787636808` is forced into the development part. The reserved evaluation part cannot be used for E rule or threshold selection; once the calibration part is used for tuning, that should also be recorded truthfully.

These samples all come from the official **train**; the above partition is only E's internal use, has not yet been coordinated with Diagnosis's future training/test use, and is not isolated by Wikipedia page, so it cannot be used directly as an unseen test set for the whole system. After the team determines the shared reserved IDs, a JSON string list can be supplied with `--reserved-ids`, with a full new set of output paths specified, to generate the next version. All IDs of the same normalised question should be excluded together; the program refuses to continue across uses. The frozen files of this version are not overwritten.

## First review batch

The development part has 19,010 screening candidates and 22,730 EM-matched records. Each is sorted by ID hash with a fixed seed, 80 candidates and 20 controls are selected, and they are arranged with one control after every four candidates. 100 is the first-batch human workload; all 74,097 records are still retained.

| File | Use |
|---|---|
| [Review TXT](../artifacts/answer_repair/nq_train_dev_v1/review/review_queue.txt) | Number, question, C1 answer and pending-review status |
| [Full review JSONL](../artifacts/answer_repair/nq_train_dev_v1/review/review_queue.jsonl) | Top-5, offline reference, metrics, source and human fields |
| [C1-compatible input](../artifacts/answer_repair/nq_train_dev_v1/inputs/nq_train_review_c1.jsonl) | E runner development input, with offline scoring fields |
| [Pure runtime input](../artifacts/answer_repair/nq_train_dev_v1/inputs/nq_train_review_runtime.jsonl) | Only sample_id, dataset, query, initial_answer, evidence |
| [Generation record](../artifacts/answer_repair/nq_train_dev_v1/preparation.json) | Counts, SHA-256 and status of the full set and review package |
| [Entry check](../artifacts/answer_repair/nq_train_dev_v1/preflight.json) | Existing v7 strategy read successfully, no model called |

The review separately judges whether C1 is correct, whether the evidence is sufficient, whether the evidence is easy to locate, and whether the reference has problems. Utilisation cannot be concluded just because gold is in Top-5, nor can EM=1 controls automatically be treated as correct and evidence-supported. See the [upstream annotation guide](NQ_TRAIN_DIAGNOSIS_ANNOTATION_GUIDE.md). Assisted judgements and user confirmations are saved separately to `data/annotations/assisted/` and `data/annotations/human/`, linked by `review_set_id + dataset + sample_id`, and the original queue is kept as frozen material. This does not mean the Diagnosis owner has completed full labelling.

This queue only reviews the initial C1 and evidence; the new [v8 output review](../artifacts/answer_repair/v8_nq75/review/changed_outputs_pending.jsonl) is saved separately, reuses the NQ Train first-batch numbering, and distinguishes strategies. It is a different task from the old [v7 output review](../artifacts/answer_repair/v7_combined_review/review_all.txt), and the numbering cannot be mixed. The first batch is used to identify natural target errors, correct controls and reference disputes, supporting v8 rule design. 80:20 is the screening ratio and cannot be used to estimate the overall utilisation ratio or whole-set accuracy; later evaluation should set up a separate queue not screened by correctness.

## Reproduction

Use Python 3.11+. Download the 15 JSONL files from Drive, keep the file names, and put them in `data/processed/external/nq_train_delivery_20261017/`. The local file name and overall hash of the original ZIP are in the audit; after repackaging the ZIP hash may differ, and the shard manifest is the basis for data consistency.

```sh
python scripts/prepare_nq_train_answer_repair.py
python -m unittest discover -s tests -q
python scripts/run_answer_repair.py --dataset nq --c1 artifacts/answer_repair/nq_train_dev_v1/inputs/nq_train_review_c1.jsonl --all --strategy short_answer_quote --check-only --out results/answer_repair/nq_train_dev_v1_preflight_only.jsonl
```

The preparation command verifies the source, converts all records, and generates the partition and review package; rerunning is allowed when the existing output is byte-identical, and overwriting is refused on any difference. 10 new tests cover gold isolation, model settings, evidence completeness, partition isolation, order invariance, duplicate IDs, corruption and overwrite protection; all 112 tests pass. The last command is only an entry check; `remaining_call_upper_bound=100` is the potential call upper bound, not the actual number of calls.

The local full file is `data/processed/answer_repair/nq_train_v1/c1_scored.jsonl`, about 283 MB, containing records for each use and the offline reference. **Do not run `--all` directly on the full file to select strategies or run development experiments**. Development first uses the fixed 100 records above; when expanding the development batch, filter by frozen ID/use and save a new manifest. The original shards, full conversion and full partition IDs are ignored by Git; code, audits, configs and small review packages are under version control.

The implementation is in the [adapter](../src/compensation/nq_train_adapter.py) and the [preparation entry point](../scripts/prepare_nq_train_answer_repair.py). Runtime fields are exported through an allowlist; gold, EM/F1, source and human labels never enter the E prompt.
