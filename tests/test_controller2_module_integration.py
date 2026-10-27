"""Merge boundaries: migrated types, real module dispatch and safe defaults.

Providers are fictional; these checks do not run GPU models or score test data.
"""
import json
from pathlib import Path
import unittest

from scripts.run_controller_integration import smoke_inputs, smoke_pipeline
from src.compensation import schema as legacy
from src.compensation import refinement_schema as current
from src.compensation.answer_repair import Generation
from src.compensation.evidence_refinement import EvidenceRefiner
from src.compensation.refinement_adapter import EvidenceRefinementAdapter
from src.controller.execution import ExecutionConfig, decision_for
from src.controller.schema import ControllerAction

ROOT = Path(__file__).resolve().parents[1]


class Controller2ModuleTests(unittest.TestCase):
    def backend(self):
        config = json.loads((ROOT / 'configs/controller/current_baseline.json').read_text())
        return smoke_pipeline(ExecutionConfig(**config['execution_config']))

    def test_old_and_new_refinement_imports_share_the_same_types(self):
        import src.compensation as public
        for name in ('RefinementConfig', 'RefinementResult', 'RefinementStatus'):
            self.assertIs(getattr(legacy, name), getattr(current, name))
            self.assertIs(getattr(public, name), getattr(current, name))
        self.assertIs(public.EvidenceRefinementAdapter, EvidenceRefinementAdapter)

    def test_legacy_accepted_result_derives_candidate_but_rejection_stays_unknown(self):
        item = smoke_inputs()[0]
        fields = dict(initial_answer=item.initial_answer, evidence=tuple(item.evidence),
                      rerank_scores=(), original_passage_ids=tuple(e.passage_id for e in item.evidence),
                      dropped_passage_ids=(), candidate_pool_size=5, regenerated=True, used_probe_hint=False)
        accepted = legacy.RefinementResult(status=current.RefinementStatus.ANSWER_REPLACED,
                                          final_answer='Amber', **fields)
        self.assertEqual(accepted.candidate_answer, accepted.final_answer)
        rejected = legacy.RefinementResult(status=current.RefinementStatus.KEPT_ORIGINAL,
                                          final_answer=item.initial_answer, reject_reason='not_grounded', **fields)
        self.assertIsNone(rejected.candidate_answer)
        with self.assertRaises(ValueError):
            legacy.RefinementResult(status=current.RefinementStatus.ANSWER_REPLACED,
                                    final_answer='Amber', candidate_answer='Crimson', **fields)

    def test_rejected_rerank_candidate_survives_the_metered_controller_trace(self):
        backend = self.backend()
        backend.raw_generator.generate = lambda *args: Generation('Crimson', 0.0)
        item = smoke_inputs()[0]
        result = backend.execute(item, decision_for(ControllerAction.EVIDENCE_REFINEMENT, item.sample_id))
        self.assertEqual(result['final_answer'], item.initial_answer)
        self.assertEqual(result['final_evidence'], [dict(rank=e.rank, passage_id=e.passage_id,
                         text=e.text, score=e.score) for e in item.evidence])
        self.assertEqual(result['module_trace']['candidate_answer'], 'Crimson')
        self.assertEqual(result['module_trace']['reject_reason'], 'not_grounded_in_kept_evidence')

    def test_pruning_changed_text_with_unchanged_ids_reaches_the_new_adapter(self):
        from src.compensation.schema import EnhancementRequest
        item = smoke_inputs()[0]
        generated = []

        class Pruner:
            def prune(self, query, texts):
                return [text.replace('The capital of Luma is', 'Luma capital:') for text in texts]

        def generate(prompt):
            generated.append(prompt)
            return 'Amber'

        refiner = EvidenceRefiner(lambda query, texts: [0.0] * len(texts), generate,
                                  prompt_builder=lambda item, evidence: '\n'.join(e.text for e in evidence),
                                  pruner=Pruner())
        request = EnhancementRequest.from_runtime(query=item.query, initial_answer=item.initial_answer,
            evidence=item.evidence, controller_decision=decision_for(ControllerAction.EVIDENCE_REFINEMENT,
                item.sample_id), sample_id=item.sample_id, dataset='nq')
        result = EvidenceRefinementAdapter(refiner).enhance(request)
        self.assertEqual(len(generated), 1)
        self.assertIn('Luma capital:', generated[0])
        self.assertTrue(result.metadata['evidence_changed'])
        self.assertEqual([e.passage_id for e in result.evidence], [e.passage_id for e in item.evidence])
        self.assertEqual(result.final_answer, 'Amber')

    def test_fever_retrieval_and_disabled_E_keep_the_registered_behavior(self):
        backend = self.backend()
        item = smoke_inputs()[1]
        retrieval = backend.execute(item, decision_for(ControllerAction.RETRIEVAL_ENHANCEMENT, item.sample_id))
        self.assertIsNone(retrieval['error'])
        self.assertEqual(retrieval['final_answer'], 'SUPPORTS')
        repair = backend.execute(item, decision_for(ControllerAction.GENERATION_REPAIR, item.sample_id))
        self.assertEqual(repair['status'], 'verification_unavailable')
        self.assertEqual(repair['final_answer'], item.initial_answer)

    def test_default_refiner_does_not_load_candidate_models_or_enable_pruning(self):
        refiner = EvidenceRefiner(lambda query, texts: [0.0] * len(texts), lambda prompt: 'Amber')
        self.assertEqual(refiner.config.reranker_model, 'BAAI/bge-reranker-large')
        self.assertEqual(refiner.config.keep_k, 5)
        self.assertIsNone(refiner.pruner)
        self.assertFalse(refiner.config.regenerate_when_unchanged)


if __name__ == '__main__':
    unittest.main()
