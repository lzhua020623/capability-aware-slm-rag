"""Keep the 1d51856 additions inside the registered data/resource boundaries."""
from contextlib import redirect_stderr,redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

import numpy as np

from src.data.team_nq import BoundaryError,TeamNQBoundary
from tests.test_team_nq_boundary import fictional

ROOT=Path(__file__).resolve().parents[1]


class LatestRerankBoundaries(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.boundary=TeamNQBoundary(ROOT)
        cls.ids=cls.boundary.allowed_ids('validation')[:3]

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.folder=Path(self.tmp.name)

    def arm(self,name='arm',ids=None):
        path=self.folder/(name+'.json')
        rows=[{'sample_id':sid,'initial_answer':'Wrong','final_answer':'Aster','gold_answer':'Aster'}
              for sid in (self.ids[:2] if ids is None else ids)]
        path.write_text(json.dumps({'rows':rows}))
        return path

    def test_selector_blocks_test_arm_before_decoding_answers_or_loading_models(self):
        from scripts import evaluate_answer_selector as mod
        path=self.folder/'test_arm.json'
        path.write_text('{"rows":[{"sample_id":"'+self.boundary.splits['test'][0]+'","gold_answer":MUST_NOT_DECODE}]}')
        with patch.object(mod,'load_arms') as reader,patch.object(mod.AutoTokenizer,'from_pretrained') as model:
            with self.assertRaises(BoundaryError):
                mod.main(['--arms','one='+str(path),'--out',str(self.folder/'out.json')])
            reader.assert_not_called();model.assert_not_called()

    def test_selector_rejects_different_cohorts_and_disagreeing_references_before_models(self):
        from scripts import evaluate_answer_selector as mod
        one=self.arm('one');two=self.arm('two',self.ids[1:])
        for mismatch in ('cohort','gold','before'):
            if mismatch!='cohort':
                two=self.arm('two');data=json.loads(two.read_text())
                data['rows'][0]['gold_answer' if mismatch=='gold' else 'initial_answer']='Different'
                two.write_text(json.dumps(data))
            with patch.object(mod.AutoTokenizer,'from_pretrained') as model:
                with self.assertRaises(BoundaryError):
                    mod.main(['--arms','one='+str(one),'two='+str(two),'--out',str(self.folder/'out.json')])
                model.assert_not_called()

    def test_selector_does_not_overwrite_existing_result(self):
        from scripts import evaluate_answer_selector as mod
        output=self.folder/'out.json';output.write_text('preserve')
        with self.assertRaises(FileExistsError):mod.main(['--arms','x=unused','--out',str(output)])
        self.assertEqual(output.read_text(),'preserve')

    def test_selector_reads_only_requested_validation_and_keeps_gold_out_of_pick(self):
        from scripts import evaluate_answer_selector as mod
        source=self.folder/'source.jsonl'
        with source.open('w') as handle:
            for sid in self.ids[:2]:handle.write(json.dumps(fictional(sid))+'\n')
            for sid in (self.ids[2],self.boundary.splits['test'][0]):
                handle.write('{"sample_id":"'+sid+'","gold":MUST_NOT_DECODE}\n')
        model=SimpleNamespace();model.to=lambda device:model;model.eval=lambda:model
        seen=[]
        def pick(*args):
            self.assertEqual(len(args),7)
            self.assertIsInstance(args[2],str);self.assertIsInstance(args[3],str)
            self.assertIsInstance(args[4],list);seen.append(args[4]);return args[4][-1]
        out=self.folder/'out.json'
        with patch.object(mod.AutoTokenizer,'from_pretrained',return_value=object()), \
             patch.object(mod.AutoModelForSequenceClassification,'from_pretrained',return_value=model), \
             patch.object(mod.torch.cuda,'is_available',return_value=False), \
             patch.object(mod,'pick',side_effect=pick),redirect_stdout(io.StringIO()):
            mod.main(['--arms','one='+str(self.arm()),'--records',str(source),'--out',str(out)])
        result=json.loads(out.read_text());self.assertEqual(result['samples'],2)
        self.assertEqual(set(result['sample_ids']),set(self.ids[:2]))
        self.assertTrue(result['data_boundary']['test_sealed'])
        self.assertFalse(result['selector_training_provenance_verified']);self.assertTrue(seen)

    def test_top_n_must_be_positive_and_frozen_backend_cannot_claim_a_wider_pool(self):
        from scripts import evaluate_refinement_end_to_end as mod
        for args in (['--top-n','50'],['--top-n','0','--backend','echo'],['--top-n','-1','--backend','echo']):
            with patch.object(mod,'build_refiner') as model,redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):mod.main(args+['--out',str(self.folder/'out.json')])
                model.assert_not_called()

    def test_candidate_registry_failure_precedes_encoder_initialization(self):
        from src.compensation.candidates import IndexCandidateProvider
        paths=[self.folder/name for name in ('corpus.jsonl','index','ids.jsonl')]
        for path in paths:path.write_text('fixture')
        with patch('src.controller.nq_train_resources.validate_resources',side_effect=ValueError('fingerprint mismatch')), \
             patch('src.compensation.candidates._native_dependencies') as dependencies:
            with self.assertRaisesRegex(ValueError,'fingerprint mismatch'):
                IndexCandidateProvider.from_frozen_artifacts(*paths)
            dependencies.assert_not_called()

    def test_candidate_loader_pins_verified_encoder_revision_and_attaches_resource_receipt(self):
        from src.compensation.candidates import IndexCandidateProvider
        paths={name:self.folder/name for name in ('corpus','index','ids')}
        paths['corpus'].write_text(json.dumps({'passage_id':'p0','text':'Fictional source'})+'\n')
        paths['ids'].write_text(json.dumps({'passage_id':'p0'})+'\n');paths['index'].write_text('fixture')
        registry={'model_name':'BAAI/bge-base-en-v1.5','verified_model_revision':'a'*40,
                  'query_instruction':'PREFIX: ','embedding_dimension':768,'corpus_size':1}
        receipt={'registered':True}
        index=type('IndexFlatIP',(),{'ntotal':1,'d':768})()
        fake_faiss=SimpleNamespace(read_index=lambda path:index)
        encoder=Mock()
        with patch('src.compensation.candidates._native_dependencies',return_value=(fake_faiss,encoder)), \
             patch('src.controller.nq_train_resources.validate_resources',return_value=(registry,paths,receipt)):
            provider=IndexCandidateProvider.from_frozen_artifacts(paths['corpus'],paths['index'],paths['ids'],top_n=3)
            encoder.assert_called_once_with(registry['model_name'],revision='a'*40,device='cpu')
            provider._encode_query('fiction')
            encoder.return_value.encode.assert_called_once_with(['PREFIX: fiction'],normalize_embeddings=True,convert_to_numpy=True)
            self.assertEqual(provider.resource_receipt,receipt)
        encoder.reset_mock()
        with patch('src.compensation.candidates._native_dependencies',return_value=(fake_faiss,encoder)), \
             patch('src.controller.nq_train_resources.validate_resources',return_value=(registry,paths,receipt)):
            alternate=self.folder/'alternate';alternate.write_text('fixture')
            with self.assertRaisesRegex(ValueError,'separate registered'):
                IndexCandidateProvider.from_frozen_artifacts(alternate,paths['index'],paths['ids'])
            encoder.assert_not_called()


if __name__=='__main__':unittest.main()
