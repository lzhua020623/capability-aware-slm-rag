"""Sealed task coverage, runtime gold isolation, and interrupted generation."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from scripts.run_controller_fever_c1 import run_collection, engineering_smoke
from src.compensation.answer_repair import Generation, MODEL_NAME, MODEL_REVISION
from src.controller.fever_c1 import (COUNTS, REQUIRED_SOURCES, generate_one, verify_collection,
                                     validate_retrieval, validate_task)
from src.generation.prompts import FEVER_PROMPT_INSTRUCTION, build_fever_rag_prompt
from src.retrieval.offset_corpus import file_sha


def row(sample_id='one'):
    return {'sample_id':sample_id,'dataset':'fever','query':'The capital is Amber.',
            'evidence':[{'rank':i,'passage_id':f'p{i}','text':f'The capital is Amber. Evidence {i}.',
                         'score':1-i/10} for i in range(1,6)]}


class C1ContractTests(unittest.TestCase):
    def test_engineering_smoke_uses_only_two_fictional_inputs(self):
        class Generator:
            def generate(self,prompt,budget):
                self_outer.assertIn('fictional',prompt);self_outer.assertEqual(budget,64)
                return Generation('SUPPORTS',.01)
        self_outer=self
        result=engineering_smoke(Generator())
        self.assertTrue(result['passed']);self.assertEqual(result['llm_calls'],2)

    def test_exact_original_prompt_budget_and_raw_observation(self):
        calls=[]
        class Generator:
            def generate(self,prompt,budget):
                calls.append((prompt,budget));return Generation(' supports ',.01,8,2)
        runtime,trace=generate_one(row(),Generator())
        self.assertEqual(calls,[(build_fever_rag_prompt(row()['query'],row()['evidence']),64)])
        self.assertEqual(runtime['initial_answer'],'supports')
        self.assertEqual(trace['parsed_label'],'SUPPORTS')
        self.assertEqual(trace['generation']['text'],' supports ')
        self.assertEqual(set(runtime),{'sample_id','dataset','query','evidence','initial_answer'})

    def test_empty_model_output_becomes_visible_unknown_never_binary_guess(self):
        class Generator:
            def generate(self,*args):return Generation('',.01)
        runtime,trace=generate_one(row(),Generator())
        self.assertEqual(runtime['initial_answer'],'UNKNOWN')
        self.assertEqual(trace['parsed_label'],'UNKNOWN')
        self.assertTrue(trace['empty_output_as_unknown'])

    def test_offline_fields_duplicate_ids_bad_ranks_and_nonfinite_scores_rejected(self):
        task={k:row()[k] for k in ('sample_id','dataset','query')}
        validate_task(task)
        for key in ('gold_answer','label','correct','initial_answer'):
            with self.assertRaises(ValueError):validate_task({**task,key:'SENTINEL'})
            with self.assertRaises(ValueError):validate_retrieval({**row(),key:'SENTINEL'})
        bad=row();bad['evidence'][1]['passage_id']='p1'
        with self.assertRaises(ValueError):validate_retrieval(bad)
        bad=row();bad['evidence'][1]['rank']=1
        with self.assertRaises(ValueError):validate_retrieval(bad)
        bad=row();bad['evidence'][1]['score']=float('nan')
        with self.assertRaises(ValueError):validate_retrieval(bad)
        bad=row();bad['evidence'][1]['score']=float('inf')
        with self.assertRaises(ValueError):validate_retrieval(bad)
        bad=row();bad['evidence'][1]['text']=' '
        with self.assertRaises(ValueError):validate_retrieval(bad)


class CollectionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.collection=self.root/'collection';self.collection.mkdir()
        self.manifest={'schema_version':'controller_fever_c1_collection_v1','counts':COUNTS,
            'model_name':MODEL_NAME,'model_revision':MODEL_REVISION,'max_new_tokens':64,
            'prompt_instruction':FEVER_PROMPT_INSTRUCTION,'source_files':{},'files':{},'ordered_ids':{},
            'gold_included':False,'generation_performed':False}
        for source in REQUIRED_SOURCES:
            p=self.root/source;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('fixture')
            self.manifest['source_files'][source]=file_sha(p)
        for role,count in COUNTS.items():
            path=self.collection/role/'retrieval.jsonl';path.parent.mkdir()
            ids=[f'{role}_{i}' for i in range(count)]
            path.write_text(''.join(json.dumps(row(i))+'\n' for i in ids))
            self.manifest['ordered_ids'][role]=ids
            self.manifest['files'][role]={'path':f'{role}/retrieval.jsonl','sha256':file_sha(path)}
        self.save_manifest()

    def save_manifest(self):
        (self.collection/'manifest.json').write_text(json.dumps(self.manifest))

    def test_all_900_fictional_tasks_are_generated_once_and_kept_separate(self):
        class Generator:
            n=0
            def generate(self,*args):self.n+=1;return Generation('SUPPORTS',.01,3,2)
        gen=Generator();out=self.root/'out'
        with contextlib.redirect_stdout(io.StringIO()):
            state=run_collection(self.collection,out,gen,root=self.root)
        self.assertEqual(gen.n,900);self.assertEqual(state['n_completed'],900)
        self.assertEqual(state['state'],'completed')
        self.assertFalse(state['quality_scored'])
        for role,count in COUNTS.items():
            self.assertEqual(len((out/role/'runtime.jsonl').read_text().splitlines()),count)
        with self.assertRaises(FileExistsError):run_collection(self.collection,out,gen,root=self.root)

    def test_source_or_data_drift_fail_before_any_generation(self):
        (self.root/'src/generation/qwen.py').write_text('changed')
        with self.assertRaisesRegex(ValueError,'source fingerprint'):verify_collection(self.collection,root=self.root)
        (self.root/'src/generation/qwen.py').write_text('fixture')
        path=self.collection/'development/retrieval.jsonl';path.write_text(path.read_text()+'\n')
        with self.assertRaisesRegex(ValueError,'retrieval fingerprint'):verify_collection(self.collection,root=self.root)

    def test_missing_or_reordered_rows_cannot_be_resealed_as_complete(self):
        path=self.collection/'development/retrieval.jsonl'
        values=path.read_text().splitlines();path.write_text('\n'.join(values[::-1])+'\n')
        self.manifest['files']['development']['sha256']=file_sha(path);self.save_manifest()
        with self.assertRaisesRegex(ValueError,'reordered'):verify_collection(self.collection,root=self.root)

    def test_provider_failure_preserves_partial_run_and_marks_failed(self):
        class Generator:
            n=0
            def generate(self,*args):
                self.n+=1
                if self.n==3:raise RuntimeError('GPU failure')
                return Generation('REFUTES',.01)
        out=self.root/'failed'
        with self.assertRaisesRegex(RuntimeError,'GPU failure'):
            run_collection(self.collection,out,Generator(),root=self.root)
        state=json.loads((out/'manifest.json').read_text())
        self.assertEqual((state['state'],state['n_completed']),('failed',2))
        self.assertEqual(len((out/'development/runtime.jsonl').read_text().splitlines()),2)
        self.assertFalse((out/'selection').exists())


if __name__ == '__main__':
    unittest.main()
