# E v6: single-pass extraction and independent MNLI verdict

This round is a fixed development experiment after the negative v5 results. The old code strategies and frozen experiment packages are kept, and two explicitly optional run strategies are added. Human acceptance of results is deferred at the user's request; automatic checks do not replace human conclusions. v6r1 only adds resume compatibility for JSON environment fields, cost metadata, and replay tools; the model, prompts, thresholds, list, and strategies are unchanged. The original v6 archive is kept, but its real GPU execution was not confirmed.

## Pre-fixed strategies

- `extractive_reanswer`, NQ only: the same Qwen2.5-7B-Instruct / NF4 / FP16 / greedy / seed 42, at most one 384-token generation. Outputs only `spans`; the program uses the validated verbatim character spans as the answer, joining multiple spans with ` ; `. At most 3 spans, each at most 80 words, at most 120 words in total. Words, negations, and punctuation are preserved; formatting differences in whitespace and case are allowed. Non-existent, overlapping, duplicate, and over-long spans are rejected. Being traceable to a citation does not mean being semantically correct; choosing the wrong event can still pass.
- `fever_nli`, FEVER only: independent `microsoft/deberta-large-mnli`, revision `7296194b9009373def4f7c5dad292651e4b5cf4e`, FP32, eval, batch 4, threshold fixed at 0.90. The claim is always the full hypothesis and each C1 evidence passage is a premise; over-long evidence is split into windows overlapping by 64 tokens, keeping all tokens, without truncating the claim. If any window has entailment ≥ 0.90 and none has contradiction ≥ 0.90, output SUPPORTS; the reverse gives REFUTES; a bidirectional high-score conflict or no high score in either direction keeps the initial answer. 0.90 is a classifier score threshold not calibrated on FEVER and does not represent 90% factual accuracy; cross-window, multi-hop inference, and same-name entities can still fail. Window source text, positions, the three class probabilities, the model version, and cost are all saved.

The official NLI model card states MNLI fine-tuning; it was chosen to avoid a model with explicit FEVER supervision introducing extra task labels. The official materials cannot prove that the pretraining corpus has no overlap at all with this project, so absolute freedom from contamination cannot be claimed.

Sources: [pinned-version model card](https://huggingface.co/microsoft/deberta-large-mnli/blob/7296194b9009373def4f7c5dad292651e4b5cf4e/README.md), [official implementation](https://github.com/microsoft/DeBERTa), [label mapping](https://huggingface.co/microsoft/deberta-large-mnli/blob/7296194b9009373def4f7c5dad292651e4b5cf4e/config.json).

NLI is an experiment that adds an independent model, not a fair replacement comparison under the same SLM budget. `extra_calls/latency/input_tokens` count only Qwen; `verifier_calls/verifier_latency/verifier_input_tokens` separately count classifier forward batches, batch wall-clock latency (including batched encoding, transfer, and forward, excluding the prior windowing), and unpadded input tokens. Model download/initialization time is not included in per-sample forward latency; full run logs are kept. `--max-new-tokens 1` is a placeholder argument for the NLI CLI; there is no generation and no tokens are produced.

## Fixed data and stopping conditions

1. Reuse the 28 reviewed training development samples of v5: FEVER 14, NQ 14. Sources, human inclusion reasons, and old EM inconsistencies are unchanged.
2. Additionally run the frozen old dev50: FEVER 50, NQ 50, to observe harmful repairs on correct samples. Overlap with batch 1 is FEVER 1, NQ 4; the 128 outputs cannot be treated as independent samples, and repairs of duplicate samples cannot be counted twice.
3. This round runs only these two pre-fixed strategies; thresholds are not repeatedly modified based on results, nor is the top score on the small development set pursued. Diagnosis validation/test is not touched.
4. Once automatic result archiving, CPU deterministic replay, Controller boundary checks, and delivery documentation are complete, the engineering work can be handed over; human semantic acceptance and the team's final held-out acceptance are listed separately as pending. This round's development results must not be used to claim overall repair effectiveness.

## Code and reproduction

- `src/compensation/extractive_repair.py`: NQ prompt and character-span backfilling.
- `src/compensation/nli_repair.py`: NLI windowing, GPU inference, conflict fallback, and offline replay.
- `src/compensation/answer_repair.py`: unified entry point; `controller_adapter.py` accepts the same strategies and still executes only upstream `generation_repair`.
- `scripts/run_answer_repair.py`: pre-checks, resume with version fingerprints, raw outputs, and cost.

```sh
python3.11 -m unittest discover -s tests
python3.11 scripts/prepare_reviewed_answer_repair.py --experiment v6 --manifest configs/splits/answer_repair_reviewed_v6.json --out-dir artifacts/answer_repair/v6r1_reviewed --bundle-name answer_repair_v6r1_reviewed_gpu
python3.11 scripts/build_reviewed_answer_repair_notebook.py --experiment v6 --bundle artifacts/answer_repair/v6r1_reviewed/answer_repair_v6r1_reviewed_gpu.tar.gz --out notebooks/history/answer_repair/answer_repair_colab_v6r1_reviewed.ipynb
```

See `ANSWER_REPAIR_RESULTS_20261014.md` for run status and conclusions; this document inside the archive is the pre-run protocol and is not rewritten with later results.
