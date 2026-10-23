"""Outcome-blind cohort selection and explicit runtime/offline separation.

Selection sees IDs, queries, provenance partitions and prior exposure only.
It never ranks by a prediction, gold value, EM, failure type or annotation.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import unicodedata

from src.compensation.nq_train_adapter import offline_reference, runtime_request
from src.diagnosis.features import content_tokens, tokenize

ROLES = ("development", "selection", "final_candidate")
OLD_ROLES = {"development": "development", "calibration": "selection",
             "reserved_evaluation": "final_candidate"}


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical_query(query: str) -> str:
    text = unicodedata.normalize("NFKC", query).casefold()
    # Strip punctuation rather than replacing it, matching the earlier variant
    # audit: apostrophe/hyphen/spacing variants share a group. Symbols remain.
    return " ".join("".join(c for c in text if not unicodedata.category(c).startswith("P")).split())


def query_key(query: str) -> str:
    return sha(canonical_query(query).encode())


def priority(dataset, sample_id, role, seed=42):
    return sha(f"controller-data-v1:{seed}:{dataset}:{role}:{sample_id}".encode())


@dataclass(frozen=True)
class Candidate:
    dataset: str
    sample_id: str
    query: str
    role: str

    def __post_init__(self):
        if self.dataset not in ("nq", "fever") or self.role not in ROLES:
            raise ValueError("Unknown dataset/partition")
        if not isinstance(self.sample_id, str) or not self.sample_id.strip():
            raise ValueError("A string sample ID is required")
        if not isinstance(self.query, str) or not canonical_query(self.query):
            raise ValueError("A nonempty query is required")


class QueryGuard:
    """Exact normalized queries and direct content-token Jaccard >= 0.60.

    The inverted index avoids comparing every candidate to every old query.
    This is a lexical check, not semantic/page-level or pretraining isolation.
    """
    def __init__(self, queries=(), threshold=0.6):
        self.threshold = threshold
        self.exact, self.terms, self.postings = set(), [], defaultdict(set)
        for query in queries:
            self.add(query)

    def add(self, query):
        key = query_key(query)
        if key in self.exact:
            return
        self.exact.add(key)
        terms = frozenset(content_tokens(tokenize(unicodedata.normalize("NFKC", query))))
        index = len(self.terms)
        self.terms.append(terms)
        for term in terms:
            self.postings[term].add(index)

    def conflicts(self, query):
        if query_key(query) in self.exact:
            return True
        terms = frozenset(content_tokens(tokenize(unicodedata.normalize("NFKC", query))))
        candidates = set().union(*(self.postings[t] for t in terms)) if terms else set()
        for index in candidates:
            previous = self.terms[index]
            if len(terms & previous) / len(terms | previous) >= self.threshold:
                return True
        return False


def select_cohorts(candidates, exposed_ids, exposed_queries, counts, *, seed=42, quarantined_ids=None):
    """Conservative quarantine of cross-role variants, then hash-order sampling.

    Final candidates are selected first, without viewing model correctness.
    All selected questions are pairwise lexically separated, including within
    a role. No outcome-dependent replacement occurs after inference.
    """
    if set(counts) != set(ROLES) or any(type(n) is not int or n < 1 for n in counts.values()):
        raise ValueError("Positive sample counts required for all three roles")
    quarantined_ids = quarantined_ids or {}
    by_group, by_dataset, seen = defaultdict(list), defaultdict(list), set()
    for candidate in candidates:
        identity = (candidate.dataset, candidate.sample_id)
        if identity in seen:
            raise ValueError(f"Duplicate candidate ID: {identity}")
        seen.add(identity)
        by_group[(candidate.dataset, query_key(candidate.query))].append(candidate)
        by_dataset[candidate.dataset].append(candidate)
    quarantine = {key for key, group in by_group.items() if len({c.role for c in group}) > 1}
    selected, summary, exclusions = {}, {}, []
    for dataset, pool in sorted(by_dataset.items()):
        guard = QueryGuard(exposed_queries.get(dataset, ()))
        buckets = defaultdict(list)
        rejected = Counter()
        for c in pool:
            reason = None
            if c.sample_id in quarantined_ids.get(dataset, set()):
                reason = "previously_flagged_query_variant"
            elif (dataset, query_key(c.query)) in quarantine:
                reason = "cross_partition_query_variant"
            elif c.sample_id in exposed_ids.get(dataset, set()):
                reason = "known_exposed_id"
            if reason:
                rejected[reason] += 1
                exclusions.append({"dataset": dataset, "sample_id": c.sample_id, "reason": reason})
            else:
                buckets[c.role].append(c)
        selected[dataset] = {}
        details = {}
        for role in ("final_candidate", "selection", "development"):
            chosen, examined, conflicts = [], 0, 0
            for c in sorted(buckets[role], key=lambda c: (priority(dataset, c.sample_id, role, seed), c.sample_id)):
                examined += 1
                if guard.conflicts(c.query):
                    conflicts += 1
                    exclusions.append({"dataset": dataset, "sample_id": c.sample_id,
                                       "reason": "lexical_conflict_with_prior_or_selected_query"})
                    continue
                chosen.append(c.sample_id)
                guard.add(c.query)
                if len(chosen) == counts[role]:
                    break
            if len(chosen) != counts[role]:
                raise ValueError(f"Insufficient eligible {dataset}/{role}: {len(chosen)} of {counts[role]}")
            selected[dataset][role] = chosen
            details[role] = {"pre_lexical_pool": len(buckets[role]), "examined_in_hash_order": examined,
                             "lexical_conflicts_skipped": conflicts, "selected": len(chosen)}
        summary[dataset] = {"source_candidates": len(pool), "pre_sampling_exclusions": dict(rejected),
                            "roles": details}
    return selected, summary, exclusions


def fever_role(query, seed=42):
    bucket = int(sha(f"controller-fever-partition-v1:{seed}:{query_key(query)}".encode()), 16) % 100
    return "development" if bucket < 70 else "selection" if bucket < 85 else "final_candidate"


def nq_outputs(row):
    # No diagnosis label or precomputed EM is copied or used.
    runtime = asdict(runtime_request(row))
    gold, _ = offline_reference(row)
    return runtime, {"dataset": "nq", "sample_id": runtime["sample_id"], "gold_answer_or_label": gold}


def fever_outputs(row):
    sample_id, query = str(row["id"]), row["claim"]
    if not isinstance(query, str) or not canonical_query(query):
        raise ValueError("Empty FEVER claim")
    if row["label"] not in ("SUPPORTS", "REFUTES"):
        raise ValueError("FEVER row is outside the binary-label scope")
    pages, evidence = row.get("gold_page_ids"), row.get("gold_evidence_text")
    if (not isinstance(pages, list) or not pages or any(not isinstance(p, str) or not p for p in pages)
            or not isinstance(evidence, list) or not evidence
            or any(not isinstance(e, str) or not e.strip() for e in evidence)):
        raise ValueError("FEVER row lacks valid offline gold evidence")
    # This is a C1-generation task, deliberately not a fake DiagnosisInput.
    return {"dataset": "fever", "sample_id": sample_id, "query": query}, {
        "dataset": "fever", "sample_id": sample_id, "gold_answer_or_label": row["label"]}


def precision_notes(n):
    return {"n": n,
            "single_accuracy_worst_case_95pct_half_width_pp": round(100 * 1.96 * math.sqrt(0.25 / n), 2),
            "paired_gain_worst_case_95pct_half_width_pp": round(100 * 1.96 / math.sqrt(n), 2),
            "paired_gain_if_discordance_20pct_half_width_pp": round(100 * 1.96 * math.sqrt(0.20 / n), 2),
            "interpretation": "Illustrative normal approximations, not observed confidence intervals or a power guarantee"}


def jsonl_bytes(records):
    return b"".join((json.dumps(r, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode()
                    for r in records)
