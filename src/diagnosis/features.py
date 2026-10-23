"""Runtime-safe Diagnosis features.

Features are computed from a single DiagnosisInput only (dataset, query,
evidence, initial_answer). sample_id is never used. No dataset-level
statistics are computed here: any scaling must be fitted on the training
split by the classifier pipeline.

Answer/evidence features describe NQ answer spans. FEVER answers are a
SUPPORTS/REFUTES label, so those features are 0 for FEVER and the FEVER
prediction is encoded by is_refutes_prediction instead.
"""

from __future__ import annotations

import re
import statistics

from src.diagnosis.schema import DatasetName, DiagnosisInput

TOKEN_RE = re.compile(r"[a-z0-9]+")

STOPWORDS: frozenset[str] = frozenset(
    """
    a an the of in on at to for from by with about as into over under than
    and or but nor so if then
    is are was were be been being am do does did doing done has have had having
    it its this that these those there here
    i you he she we they me him her us them my your his our their
    what which who whom whose when where why how
    not no can could will would shall should may might must
    s t
    """.split()
)

ABSENT_ANSWER_RANK = 6

ABSTENTION_PHRASES: tuple[str, ...] = (
    "not mentioned",
    "not stated",
    "not specified",
    "not provided",
    "not given",
    "not explicitly",
    "not enough information",
    "no information",
    "cannot be determined",
    "can't be determined",
    "cannot determine",
    "unable to determine",
    "does not mention",
    "does not specify",
    "does not provide",
    "does not contain",
    "doesn't mention",
    "doesn't specify",
    "not in the context",
    "not in the provided",
)

WH_FEATURES: tuple[tuple[str, frozenset[str]], ...] = (
    ("is_when", frozenset({"when"})),
    ("is_who", frozenset({"who", "whom", "whose"})),
    ("is_where", frozenset({"where"})),
    ("is_what", frozenset({"what"})),
)

FEATURE_NAMES: tuple[str, ...] = (
    "top1_score",
    "top2_score",
    "score_gap_1_2",
    "score_mean_top5",
    "score_std_top5",
    "score_range_top5",
    "query_coverage_top1",
    "query_coverage_max5",
    "query_coverage_mean5",
    "answer_in_top1",
    "answer_in_any",
    "answer_first_rank",
    "answer_passage_count",
    "answer_token_coverage_max",
    "answer_length_tokens",
    "answer_is_abstention",
    "is_refutes_prediction",
    "query_length_tokens",
    "evidence_total_tokens",
    "evidence_mean_tokens",
    "evidence_max_tokens",
    "is_fever",
    "is_when",
    "is_who",
    "is_where",
    "is_what",
    "is_how_many",
)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def content_tokens(tokens: list[str]) -> set[str]:
    return {token for token in tokens if token not in STOPWORDS}


def contains_sequence(haystack: list[str], needle: list[str]) -> bool:
    if not needle or len(needle) > len(haystack):
        return False
    width = len(needle)
    return any(haystack[i : i + width] == needle for i in range(len(haystack) - width + 1))


def _coverage(query_terms: set[str], passage_terms: set[str]) -> float:
    if not query_terms:
        return 0.0
    return len(query_terms & passage_terms) / len(query_terms)


def _score_features(scores: list[float]) -> dict[str, float]:
    top1 = scores[0] if scores else 0.0
    top2 = scores[1] if len(scores) > 1 else 0.0
    return {
        "top1_score": top1,
        "top2_score": top2,
        "score_gap_1_2": top1 - top2 if len(scores) > 1 else 0.0,
        "score_mean_top5": statistics.fmean(scores) if scores else 0.0,
        "score_std_top5": statistics.pstdev(scores) if scores else 0.0,
        "score_range_top5": max(scores) - min(scores) if scores else 0.0,
    }


def _answer_features(answer: str, passages: list[list[str]], passage_sets: list[set[str]]) -> dict[str, float]:
    answer_tokens = tokenize(answer)
    answer_terms = content_tokens(answer_tokens) or set(answer_tokens)
    hits = [contains_sequence(tokens, answer_tokens) for tokens in passages]
    first_rank = next((rank for rank, hit in enumerate(hits, start=1) if hit), ABSENT_ANSWER_RANK)
    normalized = " ".join(answer.lower().split())
    return {
        "answer_in_top1": float(bool(hits) and hits[0]),
        "answer_in_any": float(any(hits)),
        "answer_first_rank": float(first_rank),
        "answer_passage_count": float(sum(hits)),
        "answer_token_coverage_max": max((_coverage(answer_terms, terms) for terms in passage_sets), default=0.0),
        "answer_length_tokens": float(len(answer_tokens)),
        "answer_is_abstention": float(any(phrase in normalized for phrase in ABSTENTION_PHRASES)),
    }


def _question_type_features(query_tokens: list[str]) -> dict[str, float]:
    features = {name: 0.0 for name, _ in WH_FEATURES} | {"is_how_many": 0.0}
    if contains_sequence(query_tokens, ["how", "many"]):
        features["is_how_many"] = 1.0
        return features
    for token in query_tokens:
        for name, words in WH_FEATURES:
            if token in words:
                features[name] = 1.0
                return features
    return features


def extract_features(item: DiagnosisInput) -> dict[str, float]:
    """Return FEATURE_NAMES -> value for one runtime input."""
    evidence = sorted(item.evidence, key=lambda passage: passage.rank)
    passages = [tokenize(passage.text) for passage in evidence]
    passage_sets = [set(tokens) for tokens in passages]
    query_tokens = tokenize(item.query)
    query_terms = content_tokens(query_tokens)
    coverages = [_coverage(query_terms, terms) for terms in passage_sets]
    lengths = [len(tokens) for tokens in passages]
    is_fever = item.dataset is DatasetName.FEVER

    features: dict[str, float] = _score_features([passage.score for passage in evidence])
    features |= {
        "query_coverage_top1": coverages[0] if coverages else 0.0,
        "query_coverage_max5": max(coverages, default=0.0),
        "query_coverage_mean5": statistics.fmean(coverages) if coverages else 0.0,
    }
    if is_fever:
        features |= {name: 0.0 for name in FEATURE_NAMES if name.startswith("answer_")}
        features["is_refutes_prediction"] = float(item.initial_answer.strip().upper() == "REFUTES")
    else:
        features |= _answer_features(item.initial_answer, passages, passage_sets)
        features["is_refutes_prediction"] = 0.0
    features |= {
        "query_length_tokens": float(len(query_tokens)),
        "evidence_total_tokens": float(sum(lengths)),
        "evidence_mean_tokens": statistics.fmean(lengths) if lengths else 0.0,
        "evidence_max_tokens": float(max(lengths, default=0)),
        "is_fever": float(is_fever),
    }
    if is_fever:
        features |= {name: 0.0 for name, _ in WH_FEATURES} | {"is_how_many": 0.0}
    else:
        features |= _question_type_features(query_tokens)
    return {name: float(features[name]) for name in FEATURE_NAMES}
