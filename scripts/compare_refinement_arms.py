"""Paired comparison of two end-to-end refinement arms on the same samples.

Every arm in `results/compensation/refinement_end_to_end_*.json` runs the same
2,367 held-out accessibility samples through the same frozen generator, so two
arms differ only in the module's configuration. That makes the comparison
paired, and a paired comparison is the only honest one here: the arms share
their easy and their hopeless samples, so an unpaired difference of means
throws away most of the signal and overstates the uncertainty.

Reported per pair:

* McNemar's exact test on exact match -- of the samples where the two arms
  disagree, how lopsided is the split? This conditions on the discordant pairs
  only, which is exactly the question "does B fix things A misses more often
  than the reverse".
* A paired bootstrap confidence interval on the EM and F1 differences,
  resampling samples (not arms), so the interval reflects which questions were
  drawn rather than assuming independence between the arms.

Usage:
    python scripts/compare_refinement_arms.py top50_rerank top50_ratio0.4 top50_ratio0.6
    python scripts/compare_refinement_arms.py --baseline top50_rerank top50_ratio0.6
"""

from __future__ import annotations

import argparse
import json
import statistics
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "compensation"
DEFAULT_OUT = RESULTS / "refinement_arm_comparison.json"


def load_arm(name: str) -> dict:
    """An arm by short name (`top50_ratio0.6`) or by path."""
    path = Path(name)
    if not path.exists():
        path = RESULTS / f"refinement_end_to_end_{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"No arm at {path}. Available: {sorted(p.stem for p in RESULTS.glob('refinement_end_to_end_*.json'))}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "name": name,
        "config": payload.get("config", {}),
        "summary": payload["summary"],
        "rows": {row["sample_id"]: row for row in payload["rows"]},
    }


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two-sided exact binomial p on the discordant pairs. 1.0 when there are none."""
    total = only_a + only_b
    if total == 0:
        return 1.0
    smaller = min(only_a, only_b)
    tail = sum(comb(total, k) for k in range(smaller + 1)) / 2**total
    return min(1.0, 2 * tail)


def paired_bootstrap(pairs: list[tuple[float, float]], iterations: int, seed: int, chunk: int = 500) -> dict:
    """Percentile CI for mean(b) - mean(a), resampling the samples the arms share.

    Because the arms are paired, mean(b) - mean(a) equals the mean of the
    per-sample differences, so the bootstrap resamples that one vector instead
    of both arms -- same interval, a fraction of the work. Drawn in chunks so
    the index matrix stays small.
    """
    import numpy as np

    differences = np.array([b - a for a, b in pairs], dtype=np.float64)
    size = differences.size
    generator = np.random.default_rng(seed)
    means = np.empty(iterations, dtype=np.float64)
    done = 0
    while done < iterations:
        width = min(chunk, iterations - done)
        picks = generator.integers(0, size, size=(width, size))
        means[done : done + width] = differences[picks].mean(axis=1)
        done += width
    means.sort()
    return {
        "delta": round(float(differences.mean()), 4),
        "ci95_low": round(float(means[int(0.025 * iterations)]), 4),
        "ci95_high": round(float(means[min(iterations - 1, int(0.975 * iterations))]), 4),
    }


def compare(arm_a: dict, arm_b: dict, iterations: int, seed: int) -> dict:
    shared = sorted(set(arm_a["rows"]) & set(arm_b["rows"]))
    if not shared:
        raise ValueError(f"{arm_a['name']} and {arm_b['name']} share no samples")
    em = [(arm_a["rows"][sid]["em_after"], arm_b["rows"][sid]["em_after"]) for sid in shared]
    f1 = [(arm_a["rows"][sid]["f1_after"], arm_b["rows"][sid]["f1_after"]) for sid in shared]
    only_a = sum(1 for a, b in em if a and not b)
    only_b = sum(1 for a, b in em if b and not a)
    return {
        "baseline": arm_a["name"],
        "arm": arm_b["name"],
        "paired_samples": len(shared),
        "em_baseline": round(statistics.mean(a for a, _ in em), 4),
        "em_arm": round(statistics.mean(b for _, b in em), 4),
        "only_baseline_correct": only_a,
        "only_arm_correct": only_b,
        "mcnemar_p": mcnemar_exact(only_a, only_b),
        "em": paired_bootstrap(em, iterations, seed),
        "f1": paired_bootstrap(f1, iterations, seed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("arms", nargs="+", help="arm short names; the first is the baseline unless --baseline is given")
    parser.add_argument("--baseline", default=None)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    names = list(args.arms)
    baseline_name = args.baseline or names.pop(0)
    baseline = load_arm(baseline_name)
    rows = [compare(baseline, load_arm(name), args.iterations, args.seed) for name in names]

    payload = {
        "note": "Paired McNemar exact test on exact match plus paired bootstrap CIs, same samples through the same frozen generator.",
        "baseline": {"name": baseline["name"], "config": baseline["config"], "summary": {k: v for k, v in baseline["summary"].items() if not isinstance(v, (dict, list))}},
        "comparisons": rows,
        "iterations": args.iterations,
        "seed": args.seed,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"baseline {baseline_name}: EM {rows[0]['em_baseline']:.4f}\n")
    header = f"{'arm':<18}{'EM':>8}{'dEM':>9}{'95% CI':>20}{'dF1':>9}{'only base':>11}{'only arm':>10}{'p':>12}"
    print(header)
    for row in rows:
        ci = f"[{row['em']['ci95_low']:+.4f}, {row['em']['ci95_high']:+.4f}]"
        print(
            f"{row['arm']:<18}{row['em_arm']:>8.4f}{row['em']['delta']:>+9.4f}{ci:>20}"
            f"{row['f1']['delta']:>+9.4f}{row['only_baseline_correct']:>11}{row['only_arm_correct']:>10}{row['mcnemar_p']:>12.2e}"
        )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
