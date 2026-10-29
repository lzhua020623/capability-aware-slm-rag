# Baseline LLM Judge (Azure OpenAI)

Independent judge over **existing** baseline `predictions.jsonl`.
Does not re-run Qwen/RAG and does not rewrite predictions.

Secrets (never commit / never put in this repo):

- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_BASE_URL`
- `AZURE_OPENAI_DEPLOYMENT`

## Formal protocol (frozen before full evaluation)

1. Read predictions (preset or path).
2. If `normalized_match` is true → `auto_correct` / label `CORRECT` (**no API call**).
3. Else call Azure **Responses API** (`gpt-6.1-sol`) with only `question`, `gold`, `prediction`.
4. No `temperature` / `top_p` / `logprobs`; use `reasoning={"effort":"low"}`.
5. Labels: `CORRECT` | `INCORRECT` | `UNCERTAIN` (`UNCERTAIN` ≠ correct).
6. Rubric: whether prediction **correctly and sufficiently answers the question** (concise answers allowed when omitted detail does not change the fact / is not required by the question).
7. Output: **line 1** = exact label; optional short reason on line 2. One format retry on parse failure. Parse failure → `llm_failed` (label null), never silent `UNCERTAIN`.
8. Default is **dry-run**. Pass `--live` to call the API.

Formal full-run outputs (tracked):

- `baselines/outputs/judge/qwen_only_full/`
- `baselines/outputs/judge/rag_matched_full/`

See `baselines/README.md` for Frozen Test Results. Pilot/smoke dirs are temporary and must not be committed.

## Sync pilot / debugging

Stratified `--pilot-sample` (from `normalized_match=false` only) remains available for calibration; do not combine with `--limit`.

```bash
python baselines/judge/run_llm_judge.py --input qwen_only --pilot-sample 30 --seed 42 --dry-run
```

## Outputs schema

Under the chosen `--output-dir`:

- `judge_results.jsonl`
- `judge_summary.json`
- `run_config.json`
- `pilot_manifest.jsonl` (only when `--pilot-sample` is set)

Keys are never written to disk.
