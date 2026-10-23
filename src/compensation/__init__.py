"""Compensation components, loaded independently of neighbouring modules."""

__all__ = [
    "CrossEncoderScorer",
    "EvidenceRefiner",
    "RefinementConfig",
    "RefinementResult",
    "RefinementStatus",
    "RelevanceScorer",
    "evidence_refinement",
    "rerank_evidence",
]


def __getattr__(name):
    # Retrieval enhancement must remain usable without importing Diagnosis,
    # Probe or the refiner merely because it shares this package.
    from importlib import import_module

    modules = {
        "EvidenceRefiner": "evidence_refinement", "evidence_refinement": "evidence_refinement",
        "CrossEncoderScorer": "rerank", "RelevanceScorer": "rerank", "rerank_evidence": "rerank",
        "RefinementConfig": "schema", "RefinementResult": "schema", "RefinementStatus": "schema",
    }
    if name not in modules:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{modules[name]}"), name)
    globals()[name] = value
    return value
