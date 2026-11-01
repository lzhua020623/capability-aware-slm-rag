"""I/O helpers for baseline LLM Judge (no API calls)."""

from __future__ import annotations

import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
JUDGE_DIR = Path(__file__).resolve().parents[1] / "judge"
DEFAULT_CONFIG_PATH = JUDGE_DIR / "judge_config.json"

LABELS = ("CORRECT", "INCORRECT", "UNCERTAIN")
JUDGE_STATUSES = (
    "auto_correct",
    "llm_ok",
    "llm_failed",
    "dry_run",
    "skipped_resume",
)

# Canonical 30-sample pilot strata (scaled if --pilot-sample != 30).
PILOT_GROUP_ORDER = (
    "em0_f1_0",
    "em0_f1_ge_0_5",
    "em0_f1_mid",
    "remainder",
)
PILOT_QUOTAS_30 = {
    "em0_f1_0": 8,
    "em0_f1_ge_0_5": 8,
    "em0_f1_mid": 8,
    "remainder": 6,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_path(p: str | Path) -> Path:
    path = Path(p)
    if not path.is_absolute():
        path = ROOT / path
    return path


def load_judge_config(path: Path | None = None) -> dict:
    cfg_path = path or DEFAULT_CONFIG_PATH
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["_config_path"] = str(cfg_path)
    return cfg


def resolve_predictions_path(input_arg: str, cfg: dict) -> Path:
    """Accept preset name or filesystem path."""
    presets = cfg.get("presets") or {}
    if input_arg in presets:
        return resolve_path(presets[input_arg])
    return resolve_path(input_arg)


def default_output_dir(input_arg: str, cfg: dict, output_dir: str | None) -> Path:
    if output_dir:
        return resolve_path(output_dir)
    root = resolve_path(cfg.get("default_output_root") or "baselines/outputs/judge")
    presets = cfg.get("presets") or {}
    name = input_arg if input_arg in presets else Path(input_arg).stem
    return root / name


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_done_ids(results_path: Path) -> set[str]:
    done: set[str] = set()
    for row in read_jsonl(results_path):
        sid = row.get("sample_id")
        if sid is not None:
            done.add(str(sid))
    return done


def is_normalized_correct(row: dict) -> bool:
    val = row.get("normalized_match")
    if isinstance(val, bool):
        return val
    try:
        return int(val) == 1
    except (TypeError, ValueError):
        return False


def extract_query(row: dict) -> str:
    for key in ("query", "question"):
        if key in row and row[key] is not None:
            return str(row[key])
    return ""


def gold_as_list(gold: Any) -> list[str]:
    if gold is None:
        return []
    if isinstance(gold, str):
        return [gold] if gold.strip() else []
    if isinstance(gold, list):
        return [str(x) for x in gold]
    return [str(gold)]


def build_judge_payload(row: dict) -> dict:
    """Fields sent to the LLM only — never include EM/F1/normalized scores."""
    return {
        "question": extract_query(row),
        "gold": gold_as_list(row.get("gold")),
        "prediction": "" if row.get("prediction") is None else str(row.get("prediction")),
    }


def build_result_row(
    src: dict,
    *,
    judge_label: str | None,
    judge_reason: str,
    judge_model: str | None,
    judge_status: str,
    auto_correct: bool,
    api_attempts: int = 0,
    retry_n: int = 0,
) -> dict:
    return {
        "sample_id": src.get("sample_id"),
        "query": extract_query(src),
        "prediction": src.get("prediction"),
        "gold": src.get("gold"),
        "em": src.get("em"),
        "f1": src.get("f1"),
        "normalized_match": src.get("normalized_match"),
        "auto_correct": bool(auto_correct),
        "judge_label": judge_label,
        "judge_reason": judge_reason,
        "judge_model": judge_model,
        "judge_status": judge_status,
        "api_attempts": int(api_attempts),
        "retry_n": int(retry_n),
    }


def normalize_label(raw: str) -> str | None:
    """Accept exact label only (optionally surrounded by punctuation/whitespace)."""
    text = (raw or "").strip().upper()
    text = text.strip(" .:;-_|\"'`")
    if text in LABELS:
        return text
    return None


def parse_judge_response_text(text: str, *, max_reason_chars: int) -> tuple[str, str]:
    """Parse model text into (label, reason).

    Protocol: first non-empty line must be exactly CORRECT|INCORRECT|UNCERTAIN.
    Optional second non-empty line is a short reason. Never infer label from reason.
    Raises ValueError on failure (caller must treat as failed, not UNCERTAIN).
    """
    raw = text or ""
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("empty judge response")

    label = normalize_label(lines[0])
    if label is None:
        raise ValueError(f"cannot parse judge label from first line: {lines[0][:200]!r}")

    reason = lines[1] if len(lines) > 1 else ""
    return label, reason[:max_reason_chars]


def redact_secrets(text: str, secrets: list[str]) -> str:
    out = text
    for secret in secrets:
        if secret:
            out = out.replace(secret, "[REDACTED]")
    return out


def load_azure_env(cfg: dict) -> dict[str, str]:
    env_map = cfg.get("env") or {}
    key_name = env_map.get("api_key", "AZURE_OPENAI_API_KEY")
    url_name = env_map.get("base_url", "AZURE_OPENAI_BASE_URL")
    dep_name = env_map.get("deployment", "AZURE_OPENAI_DEPLOYMENT")
    missing = [n for n in (key_name, url_name, dep_name) if not os.environ.get(n)]
    if missing:
        raise RuntimeError(
            "Missing Azure OpenAI env vars (values never logged): " + ", ".join(missing)
        )
    return {
        "api_key": os.environ[key_name],
        "base_url": os.environ[url_name].rstrip("/"),
        "deployment": os.environ[dep_name],
    }


def summarize_results(rows: list[dict]) -> dict:
    n = len(rows)
    auto = sum(1 for r in rows if r.get("auto_correct"))
    llm_correct = sum(1 for r in rows if r.get("judge_label") == "CORRECT" and not r.get("auto_correct"))
    llm_incorrect = sum(1 for r in rows if r.get("judge_label") == "INCORRECT")
    llm_uncertain = sum(1 for r in rows if r.get("judge_label") == "UNCERTAIN")
    # api_succeeded_n / api_called_n = samples with successful judge parse (llm_ok).
    api_succeeded = sum(1 for r in rows if r.get("judge_status") == "llm_ok")
    failed = sum(1 for r in rows if r.get("judge_status") == "llm_failed")
    dry = sum(1 for r in rows if r.get("judge_status") == "dry_run")
    # Prefer explicit per-row counters when present (retry-aware runs).
    if any("api_attempts" in r for r in rows):
        api_attempted = sum(int(r.get("api_attempts") or 0) for r in rows)
        retry_n = sum(int(r.get("retry_n") or 0) for r in rows)
    else:
        # Legacy rows: one attempt per live outcome.
        api_attempted = api_succeeded + failed
        retry_n = 0
    # Judge-assisted accuracy: auto_correct OR LLM CORRECT; UNCERTAIN/null is not correct.
    assisted_correct = sum(
        1
        for r in rows
        if bool(r.get("auto_correct")) or r.get("judge_label") == "CORRECT"
    )
    return {
        "n": n,
        "auto_correct_n": auto,
        "llm_correct_n": llm_correct,
        "llm_incorrect_n": llm_incorrect,
        "llm_uncertain_n": llm_uncertain,
        "judge_assisted_accuracy": (assisted_correct / n) if n else 0.0,
        "judge_assisted_correct_n": assisted_correct,
        "api_called_n": api_succeeded,  # legacy alias of api_succeeded_n
        "api_succeeded_n": api_succeeded,
        "api_attempted_n": api_attempted,
        "retry_n": retry_n,
        "failed_n": failed,
        "dry_run_n": dry,
        "uncertain_not_counted_as_correct": True,
        "api_called_n_means": "successful_judge_parses_llm_ok_alias_of_api_succeeded_n",
        "api_succeeded_n_means": "samples_with_judge_status_llm_ok",
        "api_attempted_n_means": "total_responses_create_calls_including_retries",
        "retry_n_means": "extra_api_calls_beyond_first_per_sample",
    }


def build_judge_prompt(payload: dict, cfg: dict, *, label_only: bool = False) -> str:
    rubric = cfg.get("rubric") or {}
    golds = payload.get("gold") or []
    gold_lines = "\n".join(f"- {g}" for g in golds) if golds else "- (none)"
    notes = cfg.get("rubric_notes") or []
    notes_block = "\n".join(f"- {n}" for n in notes)
    if label_only:
        format_block = (
            "FORMAT RETRY: reply with exactly one line containing only one of:\n"
            "CORRECT\nINCORRECT\nUNCERTAIN\n"
            "No other text.\n"
        )
    else:
        format_block = (
            "Output format (strict):\n"
            "Line 1: exactly one of CORRECT / INCORRECT / UNCERTAIN\n"
            "Line 2 (optional): very short reason (<=12 words)\n"
            "Do not write JSON. Do not put the label anywhere except line 1.\n"
        )
    return (
        "You are a factual QA judge.\n"
        "Decide whether the prediction correctly and sufficiently answers the question.\n"
        "Do NOT require the prediction to reproduce every gold detail.\n"
        "Do NOT use external knowledge to correct gold; judge question + prediction + gold only.\n"
        "Labels: CORRECT | INCORRECT | UNCERTAIN.\n"
        f"CORRECT: {rubric.get('CORRECT', '')}\n"
        f"INCORRECT: {rubric.get('INCORRECT', '')}\n"
        f"UNCERTAIN: {rubric.get('UNCERTAIN', '')}\n"
        f"Notes:\n{notes_block}\n"
        f"{format_block}\n"
        f"Question: {payload.get('question', '')}\n"
        f"Gold answers:\n{gold_lines}\n"
        f"Prediction: {payload.get('prediction', '')}\n"
    )


def build_format_retry_prompt(payload: dict, cfg: dict) -> str:
    return build_judge_prompt(payload, cfg, label_only=True)


def _as_em(row: dict) -> int:
    try:
        return int(row.get("em") or 0)
    except (TypeError, ValueError):
        return 0


def _as_f1(row: dict) -> float:
    try:
        return float(row.get("f1") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def pilot_stratum(row: dict) -> str:
    """Assign a primary stratum for normalized_match=false rows."""
    em = _as_em(row)
    f1 = _as_f1(row)
    if em == 0 and f1 == 0.0:
        return "em0_f1_0"
    if em == 0 and f1 >= 0.5:
        return "em0_f1_ge_0_5"
    if em == 0 and 0.0 < f1 < 0.5:
        return "em0_f1_mid"
    return "remainder"


def scale_pilot_quotas(pilot_sample: int) -> dict[str, int]:
    """Scale canonical 8/8/8/6 quotas to sum exactly to pilot_sample."""
    if pilot_sample < 1:
        raise ValueError("pilot_sample must be >= 1")
    if pilot_sample == 30:
        return dict(PILOT_QUOTAS_30)
    total_base = sum(PILOT_QUOTAS_30.values())
    raw = {g: PILOT_QUOTAS_30[g] * pilot_sample / total_base for g in PILOT_GROUP_ORDER}
    floors = {g: int(raw[g]) for g in PILOT_GROUP_ORDER}
    remain = pilot_sample - sum(floors.values())
    # Largest remainder method (stable by group order for ties).
    frac_order = sorted(
        PILOT_GROUP_ORDER,
        key=lambda g: (raw[g] - floors[g], -PILOT_GROUP_ORDER.index(g)),
        reverse=True,
    )
    for g in frac_order:
        if remain <= 0:
            break
        floors[g] += 1
        remain -= 1
    return floors


def _stable_sample(rows: list[dict], k: int, rng: random.Random) -> list[dict]:
    if k <= 0 or not rows:
        return []
    ordered = sorted(rows, key=lambda r: str(r.get("sample_id")))
    if k >= len(ordered):
        return list(ordered)
    return rng.sample(ordered, k)


def select_stratified_pilot(
    rows: list[dict],
    *,
    pilot_sample: int = 30,
    seed: int = 42,
) -> tuple[list[dict], list[dict], dict[str, int]]:
    """Select reproducible stratified pilot from normalized_match=false only.

    Returns (selected_rows_with_pilot_group, manifest_rows, group_counts).
    If a stratum has fewer than its quota, the shortfall is filled from other
    unselected pool rows (fixed seed); those rows are labeled with the target
    quota group's ``pilot_group``.
    """
    pool = [r for r in rows if not is_normalized_correct(r)]
    if not pool:
        raise ValueError("No normalized_match=false samples available for pilot")
    if pilot_sample > len(pool):
        raise ValueError(
            f"pilot_sample={pilot_sample} exceeds available false-normalized n={len(pool)}"
        )

    quotas = scale_pilot_quotas(pilot_sample)
    rng = random.Random(seed)

    by_group: dict[str, list[dict]] = {g: [] for g in PILOT_GROUP_ORDER}
    for row in pool:
        by_group[pilot_stratum(row)].append(row)

    selected: list[dict] = []
    selected_ids: set[str] = set()
    group_counts = {g: 0 for g in PILOT_GROUP_ORDER}
    manifest: list[dict] = []

    def _available(candidates: list[dict]) -> list[dict]:
        return [r for r in candidates if str(r.get("sample_id")) not in selected_ids]

    def _take(candidates: list[dict], k: int, pilot_group: str) -> int:
        picked = _stable_sample(_available(candidates), k, rng)
        for row in picked:
            sid = str(row.get("sample_id"))
            selected_ids.add(sid)
            out = dict(row)
            out["pilot_group"] = pilot_group
            selected.append(out)
            group_counts[pilot_group] += 1
            manifest.append(
                {
                    "sample_id": row.get("sample_id"),
                    "query": extract_query(row),
                    "prediction": row.get("prediction"),
                    "gold": row.get("gold"),
                    "em": row.get("em"),
                    "f1": row.get("f1"),
                    "normalized_match": row.get("normalized_match"),
                    "pilot_group": pilot_group,
                }
            )
        return len(picked)

    def _fill_quota(group: str, need: int, primary: list[dict]) -> None:
        got = _take(primary, need, group)
        short = need - got
        if short > 0:
            _take(_available(pool), short, group)

    for group in ("em0_f1_0", "em0_f1_ge_0_5", "em0_f1_mid"):
        _fill_quota(group, quotas[group], by_group[group])

    # Remainder: prefer natural remainder stratum, then any leftover.
    rem_need = quotas["remainder"]
    got = _take(by_group["remainder"], rem_need, "remainder")
    short = rem_need - got
    if short > 0:
        _take(_available(pool), short, "remainder")

    if len(selected) != pilot_sample:
        raise RuntimeError(
            f"Pilot selection produced {len(selected)} != requested {pilot_sample}"
        )

    selected_sorted = sorted(selected, key=lambda r: str(r.get("sample_id")))
    manifest_by_id = {str(m["sample_id"]): m for m in manifest}
    manifest_sorted = [manifest_by_id[str(r["sample_id"])] for r in selected_sorted]
    counts_out = {g: group_counts[g] for g in PILOT_GROUP_ORDER}
    return selected_sorted, manifest_sorted, counts_out


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

