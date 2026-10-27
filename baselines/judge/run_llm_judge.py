#!/usr/bin/env python3
"""Azure OpenAI LLM Judge over existing baseline predictions.

Default is dry-run (no API calls). Pass --live to charge Azure.
Secrets are read only from environment variables; never logged or written.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines._lib.judge_io import (  # noqa: E402
    LABELS,
    append_jsonl,
    build_format_retry_prompt,
    build_judge_payload,
    build_judge_prompt,
    build_result_row,
    default_output_dir,
    is_normalized_correct,
    load_azure_env,
    load_done_ids,
    load_judge_config,
    parse_judge_response_text,
    read_jsonl,
    redact_secrets,
    resolve_predictions_path,
    select_stratified_pilot,
    summarize_results,
    utc_now,
    write_json,
    write_jsonl,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LLM Judge for baseline predictions (Azure OpenAI)")
    p.add_argument(
        "--input",
        required=True,
        help="Preset name (qwen_only | rag_matched_test_context_fallback) or predictions.jsonl path",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to baselines/outputs/judge/<preset_or_stem>/",
    )
    p.add_argument(
        "--config",
        default=None,
        help="Optional judge_config.json path",
    )
    p.add_argument("--limit", type=int, default=None, help="Process at most N input rows (prefix)")
    p.add_argument(
        "--pilot-sample",
        type=int,
        default=None,
        help="Stratified pilot size from normalized_match=false (e.g. 30). Mutually exclusive with --limit.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for --pilot-sample (default 42).",
    )
    p.add_argument("--resume", action="store_true", help="Skip sample_ids already in judge_results.jsonl")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not call Azure (default unless --live).",
    )
    p.add_argument(
        "--live",
        action="store_true",
        help="Actually call Azure OpenAI Responses API (incurs cost).",
    )
    return p.parse_args(argv)


def is_dry_run(args: argparse.Namespace) -> bool:
    # Safe by default: require explicit --live to spend money.
    if args.live and not args.dry_run:
        return False
    return True


def create_openai_client(api_key: str, base_url: str):
    from openai import OpenAI

    return OpenAI(api_key=api_key, base_url=base_url)


def build_responses_create_kwargs(
    *,
    deployment: str,
    prompt: str,
    max_output_tokens: int,
    reasoning_effort: str = "low",
) -> dict[str, Any]:
    """Build Responses API kwargs for gpt-6.1-sol (no sampling params).

    temperature / top_p / logprobs / top_logprobs must be absent (not null).
    GPT-6.1-sol uses reasoning effort low (not none/minimal).
    """
    return {
        "model": deployment,
        "input": prompt,
        "max_output_tokens": max_output_tokens,
        "reasoning": {"effort": reasoning_effort},
    }


def extract_response_text(resp: Any) -> str:
    text = getattr(resp, "output_text", None)
    if text:
        return str(text)
    chunks: list[str] = []
    for item in getattr(resp, "output", None) or []:
        for content in getattr(item, "content", None) or []:
            val = getattr(content, "text", None)
            if val:
                chunks.append(str(val))
    return "\n".join(chunks).strip()


def call_responses_judge(
    client: Any,
    *,
    deployment: str,
    prompt: str,
    max_output_tokens: int,
    reasoning_effort: str = "low",
) -> str:
    kwargs = build_responses_create_kwargs(
        deployment=deployment,
        prompt=prompt,
        max_output_tokens=max_output_tokens,
        reasoning_effort=reasoning_effort,
    )
    resp = client.responses.create(**kwargs)
    return extract_response_text(resp)


def _exc_status_code(exc: BaseException) -> int | None:
    for attr in ("status_code", "status"):
        val = getattr(exc, attr, None)
        if isinstance(val, int):
            return val
    resp = getattr(exc, "response", None)
    if resp is not None:
        val = getattr(resp, "status_code", None)
        if isinstance(val, int):
            return val
    return None


def is_retryable_api_error(exc: BaseException) -> bool:
    """Retry 429 / 5xx / common network failures. Do not retry other 4xx."""
    code = _exc_status_code(exc)
    if code == 429 or (code is not None and 500 <= code <= 599):
        return True
    if code is not None and 400 <= code < 500:
        return False
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    network_markers = (
        "timeout",
        "timed out",
        "connection",
        "temporarily unavailable",
        "network",
        "apiconnectionerror",
        "apietimeouterror",
        "ratelimit",
        "rate limit",
        "503",
        "502",
        "504",
        "429",
    )
    if any(m in name for m in ("timeout", "connection", "ratelimit")):
        return True
    if any(m in msg for m in network_markers):
        # Avoid treating BadRequest/unsupported-parameter as retryable via "400" alone.
        if code is None and ("badrequest" in name or "invalid_request" in msg or "unsupported parameter" in msg):
            return False
        return True
    return False


def _call_with_transport_retries(
    *,
    client: Any,
    deployment: str,
    prompt: str,
    max_tokens: int,
    reasoning_effort: str,
    max_api_retries: int,
    secrets_for_redact: list[str],
    api_attempts: int,
    retry_n: int,
) -> tuple[str | None, str | None, int, int]:
    """Returns (text, error_message, api_attempts, retry_n). text set on success."""
    transport_try = 0
    while True:
        api_attempts += 1
        if transport_try > 0:
            retry_n += 1
        try:
            text = call_responses_judge(
                client,
                deployment=deployment,
                prompt=prompt,
                max_output_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
            )
            return text, None, api_attempts, retry_n
        except Exception as exc:  # noqa: BLE001
            err = redact_secrets(f"{type(exc).__name__}: {exc}", secrets_for_redact)
            if is_retryable_api_error(exc) and transport_try < max_api_retries:
                transport_try += 1
                time.sleep(min(2 ** transport_try, 8))
                continue
            return None, err, api_attempts, retry_n


def judge_live_sample(
    *,
    client: Any,
    deployment: str,
    src: dict,
    cfg: dict,
    max_tokens: int,
    reasoning_effort: str,
    max_reason: int,
    secrets_for_redact: list[str],
) -> tuple[dict, dict[str, int]]:
    """Judge one non-auto sample with API + optional format retry.

    Returns (result_row, counters) where counters has attempted/succeeded/retry/failed.
    Parse failures are llm_failed with judge_label=None (never silent UNCERTAIN).
    """
    max_api_retries = int(cfg.get("max_api_retries") or 0)
    format_retry_budget = int(cfg.get("format_retry") or 0)
    payload = build_judge_payload(src)
    assert "em" not in payload and "f1" not in payload and "normalized_match" not in payload

    prompts = [build_judge_prompt(payload, cfg)]
    if format_retry_budget > 0:
        prompts.append(build_format_retry_prompt(payload, cfg))

    api_attempts = 0
    retry_n = 0
    last_error = "unknown"

    for prompt_idx, prompt in enumerate(prompts):
        text, err, api_attempts, retry_n = _call_with_transport_retries(
            client=client,
            deployment=deployment,
            prompt=prompt,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            max_api_retries=max_api_retries,
            secrets_for_redact=secrets_for_redact,
            api_attempts=api_attempts,
            retry_n=retry_n,
        )
        if err is not None:
            last_error = err
            # Non-retryable / exhausted transport failure: stop (do not format-retry).
            break
        assert text is not None
        try:
            label, reason = parse_judge_response_text(text, max_reason_chars=max_reason)
            out = build_result_row(
                src,
                judge_label=label,
                judge_reason=reason,
                judge_model=deployment,
                judge_status="llm_ok",
                auto_correct=False,
                api_attempts=api_attempts,
                retry_n=retry_n,
            )
            return out, {
                "attempted": api_attempts,
                "succeeded": 1,
                "retry": retry_n,
                "failed": 0,
            }
        except ValueError as exc:
            last_error = redact_secrets(f"ParseError: {exc}", secrets_for_redact)
            # Allow at most one format retry (second prompt). Count that next call as retry.
            if prompt_idx + 1 < len(prompts):
                retry_n += 1  # upcoming format-retry call is an extra attempt
                # Compensate: _call_with_transport_retries will also +1 attempt; retry_n already
                # marked for the upcoming extra call. But transport_try starts at 0 so it won't
                # add retry_n again on first create of the format prompt — we pre-incremented.
                continue
            break

    out = build_result_row(
        src,
        judge_label=None,
        judge_reason=last_error[:max_reason],
        judge_model=deployment,
        judge_status="llm_failed",
        auto_correct=False,
        api_attempts=api_attempts,
        retry_n=retry_n,
    )
    return out, {
        "attempted": api_attempts,
        "succeeded": 0,
        "retry": retry_n,
        "failed": 1,
    }


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg_path = Path(args.config) if args.config else None
    cfg = load_judge_config(cfg_path)
    pred_path = resolve_predictions_path(args.input, cfg)
    if not pred_path.is_file():
        raise SystemExit(f"Predictions not found: {pred_path}")

    out_dir = default_output_dir(args.input, cfg, args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "judge_results.jsonl"
    summary_path = out_dir / "judge_summary.json"
    run_config_path = out_dir / "run_config.json"

    dry = is_dry_run(args)
    if args.limit is not None and args.pilot_sample is not None:
        raise SystemExit("Refusing to combine --limit and --pilot-sample; choose one.")

    all_rows = read_jsonl(pred_path)
    pilot_counts: dict[str, int] | None = None
    pilot_manifest_path = out_dir / "pilot_manifest.jsonl"
    if args.pilot_sample is not None:
        if args.pilot_sample < 1:
            raise SystemExit("--pilot-sample must be >= 1")
        try:
            rows, manifest_rows, pilot_counts = select_stratified_pilot(
                all_rows,
                pilot_sample=int(args.pilot_sample),
                seed=int(args.seed),
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        write_jsonl(pilot_manifest_path, manifest_rows)
    else:
        rows = all_rows
        if args.limit is not None:
            if args.limit < 1:
                raise SystemExit("--limit must be >= 1")
            rows = rows[: args.limit]

    done = load_done_ids(results_path) if args.resume else set()
    if not args.resume and results_path.exists():
        raise SystemExit(
            f"Refusing to overwrite existing {results_path}. Pass --resume or remove the file."
        )

    max_reason = int(cfg.get("max_reason_chars") or 80)
    max_tokens = int(cfg.get("max_output_tokens") or 256)
    # GPT-6.1-sol: never send temperature (even if legacy key exists in config).
    reasoning_effort = str(cfg.get("reasoning_effort") or "low")

    client = None
    azure = None
    deployment: str | None = None
    secrets_for_redact: list[str] = []
    if not dry:
        azure = load_azure_env(cfg)
        deployment = azure["deployment"]
        secrets_for_redact = [azure["api_key"]]
        client = create_openai_client(azure["api_key"], azure["base_url"])

    write_json(
        run_config_path,
        {
            "generated_at": utc_now(),
            "input": str(pred_path),
            "output_dir": str(out_dir),
            "limit": args.limit,
            "pilot_sample": args.pilot_sample,
            "seed": int(args.seed) if args.pilot_sample is not None else None,
            "pilot_group_counts": pilot_counts,
            "pilot_manifest": str(pilot_manifest_path) if args.pilot_sample is not None else None,
            "resume": bool(args.resume),
            "dry_run": dry,
            "live": bool(args.live) and not dry,
            "judge_config": cfg.get("_config_path"),
            "labels": list(LABELS),
            "max_output_tokens": max_tokens,
            "reasoning_effort": reasoning_effort,
            "max_api_retries": int(cfg.get("max_api_retries") or 0),
            "format_retry": int(cfg.get("format_retry") or 0),
            "temperature_sent": False,
            "deployment_env": (cfg.get("env") or {}).get("deployment", "AZURE_OPENAI_DEPLOYMENT"),
            "deployment_set": bool(deployment),
            # Never write key or base URL secrets.
            "api_key_present": bool(azure and azure.get("api_key")) if azure else False,
            "note": "API key/base URL never written to disk. temperature/top_p/logprobs omitted from Responses API.",
        },
    )

    processed = 0
    api_succeeded = 0
    api_attempted = 0
    retry_total = 0
    failed = 0
    auto_n = 0

    try:
        for src in rows:
            sid = str(src.get("sample_id"))
            if sid in done:
                continue

            if is_normalized_correct(src):
                out = build_result_row(
                    src,
                    judge_label="CORRECT",
                    judge_reason="normalized_match",
                    judge_model=None,
                    judge_status="auto_correct",
                    auto_correct=True,
                    api_attempts=0,
                    retry_n=0,
                )
                append_jsonl(results_path, out)
                done.add(sid)
                auto_n += 1
                processed += 1
                continue

            if dry:
                out = build_result_row(
                    src,
                    judge_label=None,
                    judge_reason="dry_run_no_api",
                    judge_model=None,
                    judge_status="dry_run",
                    auto_correct=False,
                    api_attempts=0,
                    retry_n=0,
                )
                append_jsonl(results_path, out)
                done.add(sid)
                processed += 1
                continue

            assert client is not None and deployment is not None
            out, counters = judge_live_sample(
                client=client,
                deployment=deployment,
                src=src,
                cfg=cfg,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
                max_reason=max_reason,
                secrets_for_redact=secrets_for_redact,
            )
            api_attempted += counters["attempted"]
            api_succeeded += counters["succeeded"]
            retry_total += counters["retry"]
            failed += counters["failed"]
            append_jsonl(results_path, out)
            done.add(sid)
            processed += 1
    finally:
        # Drop client references; do not print secrets.
        client = None
        azure = None

    all_result_rows = read_jsonl(results_path)
    summary = summarize_results(all_result_rows)
    summary.update(
        {
            "input": str(pred_path),
            "output_dir": str(out_dir),
            "dry_run": dry,
            "pilot_sample": args.pilot_sample,
            "seed": int(args.seed) if args.pilot_sample is not None else None,
            "pilot_group_counts": pilot_counts,
            "this_run_processed": processed,
            "this_run_auto_correct": auto_n,
            "this_run_api_called": api_succeeded,  # legacy alias
            "this_run_api_succeeded": api_succeeded,
            "this_run_api_attempted": api_attempted,
            "this_run_retry": retry_total,
            "this_run_failed": failed,
            "labels": list(LABELS),
        }
    )
    write_json(summary_path, summary)
    print(json.dumps({"status": "JUDGE_COMPLETE", "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
