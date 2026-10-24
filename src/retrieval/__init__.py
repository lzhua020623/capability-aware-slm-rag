"""Retrieval components, loaded only when their public API is requested.

Metadata utilities also live in this package. Importing those utilities must not
require FAISS, Torch or an encoder on a generation-only worker.
"""

from importlib import import_module

__all__ = ["E5FaissRetriever", "load_index", "make_ivfpq_index"]

_EXPORT_MODULES = {
    "E5FaissRetriever": "src.retrieval.e5_faiss",
    "load_index": "src.retrieval.fever_ivfpq",
    "make_ivfpq_index": "src.retrieval.fever_ivfpq",
}


def __getattr__(name):
    if name not in _EXPORT_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORT_MODULES[name]), name)
    globals()[name] = value
    return value
