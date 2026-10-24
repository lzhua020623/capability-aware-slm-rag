"""A Qwen-only C1 worker must not require FAISS/encoders just to hash inputs."""
import subprocess
import sys
import unittest


class RetrievalImportTests(unittest.TestCase):
    def test_fresh_c1_import_does_not_load_optional_retrieval_or_model_dependencies(self):
        code='''
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'faiss','torch','sentence_transformers','transformers'}:
            raise ImportError('forbidden optional dependency: '+fullname)
sys.meta_path.insert(0,Block())
from src.controller.fever_c1 import validate_retrieval, generate_one
from src.retrieval.offset_corpus import file_sha
from src.retrieval.isolated_bge import encode_queries
assert not {'faiss','torch','sentence_transformers','transformers'} & sys.modules.keys()
print('clean C1 imports passed')
'''
        result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def test_public_retrieval_names_remain_lazy_and_cache_resolved_objects(self):
        code='''
import sys, types
import src.retrieval as package
assert package.__all__ == ['E5FaissRetriever','load_index','make_ivfpq_index']
assert 'src.retrieval.e5_faiss' not in sys.modules
fake=types.ModuleType('src.retrieval.e5_faiss')
fake.E5FaissRetriever=object()
sys.modules[fake.__name__]=fake
assert package.E5FaissRetriever is fake.E5FaissRetriever
assert package.E5FaissRetriever is package.__dict__['E5FaissRetriever']
try:
    package.not_a_public_name
except AttributeError:
    pass
else:
    raise AssertionError('Unknown export did not raise')
'''
        result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)


if __name__=='__main__':
    unittest.main()
