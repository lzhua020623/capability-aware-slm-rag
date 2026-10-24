"""FEVER dispatch, frozen prompt/fusion, fallbacks and genuine lazy lookup."""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from scripts.run_controller_integration import smoke_inputs
from src.compensation.answer_repair import Generation
from src.compensation.retrieval_enhancement.adapters import BGE_MODEL_NAME, BGE_QUERY_INSTRUCTION
from src.compensation.retrieval_enhancement.p3 import P320RetrievalEnhancer, RERANKER_MODEL, RERANKER_REVISION
from src.controller.execution import IntegratedController
from src.controller.fever_resources import DatasetRetriever, FeverFaissBackend
from src.controller.nq_train_resources import OffsetBGERetriever
from src.generation.prompts import build_fever_rag_prompt
from src.retrieval.offset_corpus import OffsetCorpus


class Scorer:
    model_name, revision = RERANKER_MODEL, RERANKER_REVISION
    def __call__(self, query, texts):
        return [float(i) for i in range(len(texts))]


class Retriever:
    supported_datasets = {'fever'}
    def __init__(self):
        self.calls = []
    def retrieve(self, query, *, dataset, top_k):
        self.calls.append((query, dataset, top_k))
        return [{'passage_id': f'extra{i}', 'text': f'Evidence text {i}.', 'score': 1-i/100}
                for i in range(top_k)]


class FeverEnhancementTests(unittest.TestCase):
    def setUp(self):
        self.item = smoke_inputs()[1]
        self.decision = {'sample_id': self.item.sample_id, 'action': 'retrieval_enhancement'}
        self.retriever, self.prompts = Retriever(), []

    def run_module(self, answer='SUPPORTS', **kwargs):
        def generate(prompt):
            self.prompts.append(prompt)
            if isinstance(answer, Exception):
                raise answer
            return answer
        return P320RetrievalEnhancer(self.retriever, Scorer(), generate, **kwargs).run(self.item, self.decision)

    def test_default_nq_api_still_rejects_fever_without_calls(self):
        result = self.run_module()
        self.assertEqual(result['status'], 'unsupported_dataset')
        self.assertFalse(self.retriever.calls or self.prompts)

    def test_explicit_fever_path_uses_original_claim_fusion_and_frozen_prompt(self):
        result = self.run_module('The label is supports.', enable_fever=True)
        self.assertEqual(self.retriever.calls, [(self.item.query, 'fever', 20)])
        self.assertEqual(self.prompts, [build_fever_rag_prompt(self.item.query, result['evidence'])])
        self.assertEqual(result['final_answer'], 'SUPPORTS')
        self.assertEqual(result['status'], 'answer_replaced')
        self.assertEqual(result['scored_pairs'], 20)
        self.assertEqual(result['rrf_k'], 60)
        self.assertFalse(result['rewriting_enabled'])
        # Label need not appear literally in the selected passages.
        self.assertNotIn('SUPPORTS', ' '.join(e['text'] for e in result['evidence']))

    def test_unparsable_label_and_provider_failure_retain_stage_input(self):
        for answer, status in [('unknown', 'kept_original'), (RuntimeError('failed'), 'generation_failed')]:
            with self.subTest(answer=str(answer)):
                result = self.run_module(answer, enable_fever=True)
                self.assertEqual(result['status'], status)
                self.assertEqual(result['final_answer'], self.item.initial_answer)
                self.assertEqual(result['generation_calls'], 1)

    def test_integrated_path_has_exactly_one_retrieval_scoring_generation(self):
        generator = SimpleNamespace(generate=lambda prompt, max_new_tokens: Generation('SUPPORTS', .01, 3, 2))
        p = IntegratedController(None, generator, {}, Scorer(), self.retriever)
        row = p.run(self.item, mode='always_retrieve')
        self.assertEqual(row['final_answer'], 'SUPPORTS')
        self.assertEqual(tuple(row['cost'][k] for k in ('retrieval_calls', 'scoring_calls', 'llm_calls')), (1, 1, 1))
        self.assertEqual(row['input'], asdict(self.item))
        self.assertNotEqual(row['final_evidence'], row['input']['evidence'])
        json.dumps(row, allow_nan=False)


class FeverBackendTests(unittest.TestCase):
    def test_actual_offset_reads_and_strict_dataset_index_guards(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            body, ids, offsets = [folder/n for n in ('body', 'ids', 'offsets.npy')]
            lines = [json.dumps({'passage_id': f'p{i}', 'text': f'text {i}', 'gold': 'SENTINEL'})+'\n'
                     for i in range(3)]
            body.write_text(''.join(lines)); ids.write_text(''.join(json.dumps({'passage_id':f'p{i}'})+'\n' for i in range(3)))
            np.save(offsets, np.array([0, len(lines[0]), len(lines[0])+len(lines[1])], dtype=np.int64))
            cls = type('IndexIVFPQ', (), {})
            index = cls()
            for k,v in dict(ntotal=3,d=768,nlist=4096,nprobe=16,metric_type=0,is_trained=True,by_residual=False).items():
                setattr(index,k,v)
            index.pq = SimpleNamespace(M=64, nbits=8)
            index.search = lambda x,k: (np.array([[.9,.8,.7]]),np.array([[2,0,1]]))
            vector=np.zeros((1,768),dtype=np.float32);vector[0,0]=1
            model=SimpleNamespace(encode=lambda *a,**k: vector)
            with OffsetCorpus(body, ids, offsets) as store:
                dense=OffsetBGERetriever(model,index,store,lambda x: None)
                backend=FeverFaissBackend(dense,expected_count=3)
                dispatch=DatasetRetriever({'fever':backend})
                hits=dispatch.retrieve('Claim',dataset='fever',top_k=3)
                self.assertEqual([r['passage_id'] for r in hits],['p2','p0','p1'])
                self.assertNotIn('SENTINEL',json.dumps(hits))
                with self.assertRaises(ValueError):
                    dispatch.retrieve('Q',dataset='nq',top_k=3)
                index.nprobe=8
                with self.assertRaisesRegex(ValueError,'nprobe'):
                    backend.retrieve('C',dataset='fever',top_k=3)


if __name__ == '__main__':
    unittest.main()
