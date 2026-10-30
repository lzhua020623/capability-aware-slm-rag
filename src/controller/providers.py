"""Resource ownership for integrated runs; never imported by CPU smoke paths."""

from __future__ import annotations

from copy import copy
from pathlib import Path

from src.compensation.answer_repair import LINE_OUTPUT_STRATEGIES
from src.compensation.retrieval_enhancement.p3 import RERANKER_MODEL, RERANKER_REVISION
from src.controller.execution import IntegratedController
from src.controller.runtime import configure_generation_runtime


def shared_repair_views(generator, config):
    """Separate stop settings, identical tokenizer/model objects; no second load."""
    views = {}
    constrained = None
    for dataset, strategy in (("nq", config.nq_repair_strategy), ("fever", config.fever_repair_strategy)):
        if strategy == 'evidence_id_v2a':
            if constrained is None:
                from src.compensation.qwen_v2a import QwenV2AGenerator
                constrained = QwenV2AGenerator.from_shared(generator)
            views[dataset] = constrained
            continue
        view = copy(generator)
        view.json_output = strategy not in LINE_OUTPUT_STRATEGIES
        views[dataset] = view
    return views


def load_pipeline(config, classifier_path, *, nq_index=None, nq_metadata=None, nq_resources=None,
                  fever_resources=None, p3_bm25=None):
    if nq_resources and (nq_index or nq_metadata):
        raise ValueError("Choose NQ Train resources or the legacy index/metadata pair")
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
    configure_generation_runtime(generator, config.prefill_chunk_size)
    # P3-20's frozen CPU/FP32/raw-logit contract. The same scorer serves reranking.
    encoder = CrossEncoder(RERANKER_MODEL, revision=RERANKER_REVISION, device="cpu",
                           max_length=512, activation_fn=torch.nn.Identity(),
                           model_kwargs={"torch_dtype": torch.float32})
    scorer = CrossEncoderScorer(lambda pairs: encoder.predict(pairs, batch_size=4, show_progress_bar=False),
                                model_name=RERANKER_MODEL)
    scorer.revision = RERANKER_REVISION
    retriever = None
    resource_receipt = None
    if nq_resources:
        from src.controller.nq_train_resources import load_backend
        retriever, resource_receipt = load_backend(nq_resources, root=Path(__file__).resolve().parents[2])
    elif nq_index:
        from src.retrieval.e5_faiss import E5FaissRetriever
        dense = E5FaissRetriever(BGE_MODEL_NAME, device="cpu", passage_prefix="",
                                query_prefix=BGE_QUERY_INSTRUCTION)
        dense.load(Path(nq_index), Path(nq_metadata))
        retriever = NQFaissBackend(dense, max_top_k=20)
    bm25 = None
    if p3_bm25:
        if not nq_resources or fever_resources:
            raise ValueError("P3-BM25 needs the NQ Train backend alone")
        from src.compensation.retrieval_enhancement.p3_bm25 import load_resources
        bm25 = load_resources(p3_bm25, retriever, Path(__file__).resolve().parents[2])
    fever_receipt = None
    if fever_resources:
        from src.controller.fever_resources import DatasetRetriever, load_backend as load_fever
        fever_backend, fever_receipt = load_fever(fever_resources, root=Path(__file__).resolve().parents[2])
        backends = {'fever': fever_backend}
        if retriever is not None:
            backends['nq'] = retriever
        retriever = DatasetRetriever(backends)
    pipeline = IntegratedController(classifier, generator, shared_repair_views(generator, config),
                                    scorer, retriever, config=config, p3_bm25=bm25)
    environment = {**generator.runtime_environment,
                   "shared_qwen_instances": 1, "reranker_model": RERANKER_MODEL,
                   "reranker_revision": RERANKER_REVISION,
                   "reranker_device": "cpu", "reranker_dtype": "float32",
                   "reranker_activation": "Identity", "reranker_max_length": 512,
                   "nq_index": str(nq_index) if nq_index else None,
                   "nq_metadata": str(nq_metadata) if nq_metadata else None,
                   "nq_train_resources": resource_receipt, 'fever_resources': fever_receipt}
    if bm25 is not None:
        environment['p3_bm25'] = {**p3_bm25, 'bm25_receipt': bm25.receipt}
    if any(s == 'evidence_id_v2a' for s in (config.nq_repair_strategy, config.fever_repair_strategy)):
        environment['E_v2a_runtime'] = next(v.runtime_environment for v in pipeline.repair_generators.values()
                                          if hasattr(v, 'compiler'))
    return pipeline, environment
