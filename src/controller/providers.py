"""Resource ownership for integrated runs; never imported by CPU smoke paths."""

from __future__ import annotations

from copy import copy
from pathlib import Path

from src.compensation.answer_repair import LINE_OUTPUT_STRATEGIES
from src.compensation.retrieval_enhancement.p3 import RERANKER_MODEL, RERANKER_REVISION
from src.controller.execution import IntegratedController


def shared_repair_views(generator, config):
    """Separate stop settings, identical tokenizer/model objects; no second load."""
    views = {}
    for dataset, strategy in (("nq", config.nq_repair_strategy), ("fever", config.fever_repair_strategy)):
        view = copy(generator)
        view.json_output = strategy not in LINE_OUTPUT_STRATEGIES
        views[dataset] = view
    return views


def load_pipeline(config, classifier_path, *, nq_index=None, nq_metadata=None):
    import torch
    from sentence_transformers import CrossEncoder
    from src.compensation.qwen_repair import QwenRepairGenerator
    from src.compensation.rerank import CrossEncoderScorer
    from src.compensation.retrieval_enhancement.adapters import (
        BGE_MODEL_NAME, BGE_QUERY_INSTRUCTION, NQFaissBackend,
    )
    from src.diagnosis.hierarchical_classifier import HierarchicalDiagnosisClassifier

    if bool(nq_index) != bool(nq_metadata):
        raise ValueError("Provide both the matching NQ index and metadata")
    classifier = HierarchicalDiagnosisClassifier.load(Path(classifier_path))
    # Load the shared Qwen once. Only the stopping adapter differs for E.
    generator = QwenRepairGenerator(json_output=False)
    # P3-20's frozen CPU/FP32/raw-logit contract. The same scorer serves reranking.
    encoder = CrossEncoder(RERANKER_MODEL, revision=RERANKER_REVISION, device="cpu",
                           max_length=512, activation_fn=torch.nn.Identity(),
                           model_kwargs={"torch_dtype": torch.float32})
    scorer = CrossEncoderScorer(lambda pairs: encoder.predict(pairs, show_progress_bar=False),
                                model_name=RERANKER_MODEL)
    scorer.revision = RERANKER_REVISION
    retriever = None
    if nq_index:
        from src.retrieval.e5_faiss import E5FaissRetriever
        dense = E5FaissRetriever(BGE_MODEL_NAME, device="cpu", passage_prefix="",
                                query_prefix=BGE_QUERY_INSTRUCTION)
        dense.load(Path(nq_index), Path(nq_metadata))
        retriever = NQFaissBackend(dense, max_top_k=20)
    pipeline = IntegratedController(classifier, generator, shared_repair_views(generator, config),
                                    scorer, retriever, config=config)
    environment = {**generator.runtime_environment,
                   "shared_qwen_instances": 1, "reranker_model": RERANKER_MODEL,
                   "reranker_revision": RERANKER_REVISION,
                   "reranker_device": "cpu", "reranker_dtype": "float32",
                   "reranker_activation": "Identity", "reranker_max_length": 512,
                   "nq_index": str(nq_index) if nq_index else None,
                   "nq_metadata": str(nq_metadata) if nq_metadata else None}
    return pipeline, environment
