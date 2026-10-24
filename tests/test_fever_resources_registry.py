import json
from pathlib import Path
import tempfile
import unittest

from src.controller.fever_resources import validate_resources
from src.retrieval.offset_corpus import file_sha
from src.compensation.retrieval_enhancement.adapters import BGE_QUERY_INSTRUCTION


class FeverRegistryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.resources=self.root/'resources';self.resources.mkdir()
        names={'corpus':'fever_corpus.jsonl','ids':'fever_bge_ivfpq_ids.jsonl',
               'index':'fever_bge_ivfpq.index','metadata':'fever_bge_ivfpq_meta.json','offsets':'fever_bge_ivfpq_offsets.npy'}
        self.paths={}
        imported={'passed':True,'mapping':{'records_checked':5396106,'full_order_verified':True,'fingerprints':{}},'extracted':{}}
        for key,name in names.items():
            path=self.resources/name;path.write_bytes(f'fixture {key}'.encode());self.paths[key]=path
            info=self.info(path)
            if key=='offsets':imported['mapping']['fingerprints']['offsets']=info
            else:imported['extracted'][name]=info
        for key,name in [('import_report','import.json'),('structure_report','structure.json'),('query_replay','replay.json')]:
            self.paths[key]=self.root/name
        self.paths['import_report'].write_text(json.dumps(imported))
        self.paths['structure_report'].write_text(json.dumps({'passed':True,'import_report_sha256':file_sha(self.paths['import_report'])}))
        replay={'passed':True,'n_queries':10,'same_ordered_top5_count':10,'encoding':{'revision':'a'*40},
                'import_report_sha256':file_sha(self.paths['import_report']),
                'structure_report_sha256':file_sha(self.paths['structure_report'])}
        self.paths['query_replay'].write_text(json.dumps(replay))
        self.registry={'schema_version':'controller_fever_resources_v1','dataset':'fever',
            'model_name':'BAAI/bge-base-en-v1.5','query_instruction':BGE_QUERY_INSTRUCTION,
            'passage_prefix':'','corpus_size':5396106,'verified_model_revision':'a'*40,
            'files':{k:self.info(p) for k,p in self.paths.items()}}
        self.registry_path=self.root/'registry.json';self.save()

    def info(self,path):
        return {'path':str(path.relative_to(self.root)),'bytes':path.stat().st_size,'sha256':file_sha(path)}

    def save(self):self.registry_path.write_text(json.dumps(self.registry))

    def test_chained_import_structure_and_query_replay_are_verified(self):
        _,paths,receipt=validate_resources(self.registry_path,root=self.root)
        self.assertEqual(paths,self.paths)
        self.assertEqual(receipt['verified_model_revision'],'a'*40)
        self.assertIsNone(receipt['historical_model_revision'])

    def test_resource_drift_even_with_rehashed_registry_is_caught_by_original_import(self):
        self.paths['corpus'].write_bytes(b'different corpus')
        with self.assertRaisesRegex(ValueError,'fingerprint'):validate_resources(self.registry_path,root=self.root)
        self.registry['files']['corpus']=self.info(self.paths['corpus']);self.save()
        with self.assertRaisesRegex(ValueError,'fingerprint'):validate_resources(self.registry_path,root=self.root)

    def test_failed_query_replay_or_different_revision_cannot_authorize_model(self):
        self.registry['verified_model_revision']='b'*40;self.save()
        with self.assertRaisesRegex(ValueError,'query replay'):validate_resources(self.registry_path,root=self.root)
        self.registry['verified_model_revision']='a'*40
        p=self.paths['query_replay'];replay=json.loads(p.read_text());replay['same_ordered_top5_count']=9
        p.write_text(json.dumps(replay));self.registry['files']['query_replay']=self.info(p);self.save()
        with self.assertRaisesRegex(ValueError,'query replay'):validate_resources(self.registry_path,root=self.root)

    def test_unknown_dataset_and_missing_bindings_are_rejected(self):
        self.registry['dataset']='nq';self.save()
        with self.assertRaises(ValueError):validate_resources(self.registry_path,root=self.root)
        self.registry['dataset']='fever';self.registry['files'].pop('query_replay');self.save()
        with self.assertRaises(ValueError):validate_resources(self.registry_path,root=self.root)


if __name__ == '__main__':
    unittest.main()
