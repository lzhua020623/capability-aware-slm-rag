"""Lexical anchors for E v7: a cited window must mention the queried entity.

This is a deterministic, model-free guard against the v6r1 failure in which a
verbatim citation about a different entity (a 2015 film called Black, a film
called Henry's Crime) drove a FEVER label change. Anchor coverage measures only
whether the claim's capitalised names and numbers appear in a window; it does
not establish entailment, relevance or correctness, and it is deliberately
conservative: a window that fails the guard cannot change the answer.

FEVER claims carry proper nouns, so anchors are capitalised tokens and numbers.
NQ questions are lower-cased, so anchors are content words; that rule is weak
by design and its effect must be measured, not assumed.
"""
from __future__ import annotations

import re

TOKEN = re.compile(r"\w+")
ANCHOR_VERSION = "v7_lexical_subject_anchor"
FEVER_MIN_COVERAGE = 0.5   # reported only; the FEVER guard is the subject chunk below
NQ_MIN_COVERAGE = 1 / 3    # at least a third of the content words (minimal rule)

STOPWORDS = frozenset("""
a an the and or but nor of in on at to for from by with without into onto over under
about above below between through during before after as is are was were be been being
am do does did doing have has had having it its this that these those there here
he she they them his her their him we us our you your i me my who whom whose which what
when where why how not no yes only also still yet ever never just than then so such
some any all both each every few more most other another very can could may might
must shall should will would part half none many several
""".split())
# Function words, pronouns, auxiliaries and quantifiers only. Adjectives that
# open names (New Jersey, First Blood, Old Trafford) are deliberately absent.


def tokens(text: str) -> list[str]:
    return TOKEN.findall(text)


def claim_anchors(claim: str) -> list[str]:
    """Capitalised tokens and numbers of a FEVER claim, stopwords removed, order kept."""
    seen, anchors = set(), []
    for token in tokens(claim):
        low = token.casefold()
        if low in STOPWORDS or (len(low) < 2 and not low.isdigit()):
            continue
        if not (token[0].isupper() or low.isdigit()):
            continue
        if low not in seen:
            seen.add(low)
            anchors.append(low)
    return anchors


def subject_chunk(claim: str) -> list[str]:
    """First run of capitalised tokens (leading stopwords dropped, numbers excluded).

    FEVER claims name their subject first ("Henry Cavill starred ...", "The
    Sugarland Express was ..."). A window that does not mention that chunk in
    full is about something else, however high its NLI score. Later anchors
    are often the disputed attribute ("Australian", "1970") and must not be
    required, because refuting evidence states a different value for them.
    """
    chunk: list[str] = []
    for token in tokens(claim):
        low = token.casefold()
        if len(low) == 1 and not low.isdigit():
            continue                      # possessive s, initials such as N' are transparent
        capitalised = token[0].isupper() and not low.isdigit()
        if capitalised and low in STOPWORDS and not chunk:
            continue                      # leading "The", "A", "In"
        if capitalised and low not in STOPWORDS:
            chunk.append(low)
            continue
        if chunk:
            break                         # the run ended
    return chunk


def question_anchors(question: str) -> list[str]:
    """Content words of a lower-cased NQ question (length >= 3, stopwords removed)."""
    seen, anchors = set(), []
    for token in tokens(question):
        low = token.casefold()
        if low in STOPWORDS or len(low) < 3:
            continue
        if low not in seen:
            seen.add(low)
            anchors.append(low)
    return anchors


def _present(anchor: str, vocabulary: set[str]) -> bool:
    if anchor in vocabulary:
        return True
    # Singular/plural only; no stemming beyond a final s.
    if anchor.endswith("s") and len(anchor) > 3 and anchor[:-1] in vocabulary:
        return True
    return anchor + "s" in vocabulary


def anchor_coverage(anchors: list[str], text: str) -> dict:
    """Which anchors occur in text; coverage is present/total (None without anchors)."""
    vocabulary = {t.casefold() for t in tokens(text)}
    present = [a for a in anchors if _present(a, vocabulary)]
    missing = [a for a in anchors if a not in present]
    coverage = len(present) / len(anchors) if anchors else None
    return {"anchors": list(anchors), "present": present, "missing": missing, "coverage": coverage}


FEVER_POLICY = "fever:subject_chunk_fully_present"


def fever_window_anchored(claim: str, window_text: str) -> dict:
    """Anchored iff every token of the claim's subject chunk occurs in the window."""
    report = anchor_coverage(claim_anchors(claim), window_text)
    subject = subject_chunk(claim)
    vocabulary = {t.casefold() for t in tokens(window_text)}
    report["subject"] = subject
    report["subject_missing"] = [a for a in subject if not _present(a, vocabulary)]
    report["anchored"] = bool(subject) and not report["subject_missing"]
    report["policy"] = f"{ANCHOR_VERSION}:{FEVER_POLICY}"
    return report


def nq_passage_anchored(question: str, passage_text: str) -> dict:
    report = anchor_coverage(question_anchors(question), passage_text)
    report["anchored"] = (report["coverage"] is not None and bool(report["present"])
                          and report["coverage"] >= NQ_MIN_COVERAGE)
    report["policy"] = f"{ANCHOR_VERSION}:nq:coverage>={NQ_MIN_COVERAGE:.4f}"
    return report
