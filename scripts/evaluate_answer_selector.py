"""Does the trained selector pick better answers than the shipped grounding gate?

The end-to-end arms leave several candidate answers per sample: the generator's
first answer plus one per arm. At least one of them is correct for 23.9% of the
held-out samples while the shipped gate delivers ~11%, so the gap is a
selection problem. This scores what the selector recovers from it.

Two settings are reported:

* **two-way** -- the drop-in replacement for the acceptance gate: choose
  between the original answer and one arm's new answer.
* **all-candidates** -- choose among the original answer and every arm's, the
  setting the oracle bound was computed in.

Every candidate is scored against the *frozen Top-5* evidence, not the evidence
its own arm happened to produce, so no candidate is judged on a context the
others never saw.

Baselines printed alongside: keep the original, each arm on its own, majority
vote, and the oracle upper bound.

Usage:
    python scripts/evaluate_answer_selector.py \
        --arms trained=results/compensation/refinement_end_to_end_accessibility.json \
               shuffle=results/compensation/refinement_end_to_end_shuffle_control.json
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_answer_selection import (  # noqa: E402
    best_candidate,
    load_arms,
    majority_vote,
    score,
    unique_candidates,
)
from scripts.train_answer_selector import ANSWER_MARKER  # noqa: E402

DEFAULT_MODEL = ROOT / "models" / "selector" / "answer_selector_team_v1"
DEFAULT_RECORDS = ROOT / "data" / "processed" / "external" / "nq_train_delivery_20261017"
DEFAULT_OUT = ROOT / "results" / "compensation" / "team_nq_v1" / "answer_selector_evaluation.json"


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def selected_records(boundary, source, wanted, role):
    """Filter raw lines by ID first; unrelated validation/test stays undecoded."""
    from src.data.team_nq import BoundaryError, ID_FIELD
    boundary.require(wanted,role)
    with tempfile.TemporaryDirectory(prefix='selector-subset-') as directory:
        subset = Path(directory)/'source.jsonl'
        with subset.open('wb') as handle:
            for line in boundary.source_lines(source):
                if not line.strip():
                    continue
                matches = ID_FIELD.findall(line)
                if len(matches) != 1 or matches[0].decode('ascii') not in boundary.membership:
                    raise BoundaryError('Source requires one registered exact sample ID')
                if matches[0].decode('ascii') in wanted:
                    handle.write(line)
        for _,record in boundary.iter_role_records(subset,roles=(role,)):
            yield record


@torch.inference_mode()
def pick(model, tokenizer, query: str, evidence: str, candidates: list[str], device, max_length: int) -> str:
    """The candidate the selector scores highest for this query and evidence."""
    if len(candidates) == 1:
        return candidates[0]
    encoded = tokenizer(
        [f"{query}{ANSWER_MARKER}{candidate}" for candidate in candidates],
        [evidence] * len(candidates),
        padding=True,
        truncation="only_second",
        max_length=max_length,
        return_tensors="pt",
    ).to(device)
    with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
        scores = model(**encoded).logits.squeeze(-1).float().reshape(-1).tolist()
    return candidates[max(range(len(candidates)), key=lambda index: scores[index])]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arms", nargs="+", required=True, help="name=path entries, saved with --save-rows")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument('--data-role', choices=('train','validation'), default='validation')
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    if args.max_length < 1 or (args.limit is not None and args.limit < 1):
        parser.error('max-length and any explicit limit must be positive')

    from src.data.team_nq import BoundaryError, TeamNQBoundary, ids_only
    if args.out.exists():
        raise FileExistsError('Preserve the historical selector evaluation; choose a new output')
    boundary = TeamNQBoundary(ROOT)
    incoming, names = {}, set()
    for spec in args.arms:
        name, separator, path = spec.partition('=')
        if not separator or not name or not path or name in names:
            raise BoundaryError('Each candidate arm needs a unique name=path')
        names.add(name)
        ids = ids_only(path)
        if not ids:
            raise BoundaryError('Candidate arm has no exact sample IDs')
        boundary.require(ids,args.data_role)  # before answer/gold JSON decoding
        incoming[name] = ids

    arms = load_arms(args.arms)
    cohort = next(iter(incoming.values()))
    if any(set(rows) != incoming[name] or set(rows) != cohort for name,rows in arms.items()):
        raise BoundaryError('Candidate arms must have identical registered ID cohorts')
    ids = sorted(cohort)
    if args.limit:
        ids = ids[: args.limit]
    reference = next(iter(arms.values()))
    gold = {key: reference[key]["gold_answer"] for key in ids}
    before = {key: reference[key]["initial_answer"] for key in ids}
    if any(rows[key]['initial_answer'] != before[key] or rows[key]['gold_answer'] != gold[key]
           for rows in arms.values() for key in ids):
        raise BoundaryError('Candidate arms disagree on original answers or scoring references')

    # The end-to-end rows carry answers only, so the query and evidence come from the records.
    evidence, queries = {}, {}
    wanted = set(ids)
    for record in selected_records(boundary,args.records,wanted,args.data_role):
        key = record['sample_id']
        if key in wanted:
            if (record.get('annotation_reference') or {}).get('gold_answer') != gold[key]:
                raise BoundaryError('Candidate reference differs from the registered source record')
            evidence[key] = ' '.join(passage['text'] for passage in record['evidence'])
            queries[key] = record['query']
    missing = [key for key in ids if key not in evidence]
    if missing:
        raise SystemExit(f"no evidence for {len(missing)} samples; wrong --records file?")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(args.model))
    model = AutoModelForSequenceClassification.from_pretrained(str(args.model), num_labels=1).to(device).eval()
    print(f"{len(ids)} samples | selector {args.model} on {device}")

    results = {"baseline_before": score(before, gold)}
    for name, rows in arms.items():
        results[f"arm_{name}"] = score({key: rows[key]["final_answer"] for key in ids}, gold)

    all_candidates = {key: unique_candidates([before[key]] + [rows[key]["final_answer"] for rows in arms.values()]) for key in ids}
    results["majority_vote"] = score(
        {key: majority_vote([rows[key]["final_answer"] for rows in arms.values()], before[key]) for key in ids}, gold
    )
    results["oracle_all_candidates"] = score({key: best_candidate(all_candidates[key], gold[key]) for key in ids}, gold)

    for name, rows in arms.items():
        chosen = {}
        for index, key in enumerate(ids, start=1):
            pair = unique_candidates([before[key], rows[key]["final_answer"]])
            chosen[key] = pick(model, tokenizer, queries[key], evidence[key], pair, device, args.max_length)
            if index % 500 == 0:
                print(f"  two-way {name}: {index}/{len(ids)}")
        results[f"selector_two_way_{name}"] = score(chosen, gold)
        results[f"oracle_two_way_{name}"] = score(
            {key: best_candidate(unique_candidates([before[key], rows[key]["final_answer"]]), gold[key]) for key in ids},
            gold,
        )

    chosen_all = {}
    for index, key in enumerate(ids, start=1):
        chosen_all[key] = pick(model, tokenizer, queries[key], evidence[key], all_candidates[key], device, args.max_length)
        if index % 500 == 0:
            print(f"  all-candidates: {index}/{len(ids)}")
    results["selector_all_candidates"] = score(chosen_all, gold)

    payload = {
        "note": "Offline re-scoring of answers that already existed; the selector adds no generation. "
        "Every candidate is judged against the frozen Top-5 evidence.",
        "selector": str(args.model),
        "data_boundary": boundary.receipt(),
        "data_role": args.data_role,
        "selector_training_provenance_verified": False,
        "model_promotion": "not certified by this evaluation; training usage IDs still required",
        "arms": {name: spec.split("=", 1)[1] for name, spec in zip(arms, args.arms)},
        "samples": len(ids),
        "sample_ids": ids,
        "exact_match": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    width = max(len(key) for key in results)
    for key, value in results.items():
        print(f"  {key:<{width}}  {value:.4f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
