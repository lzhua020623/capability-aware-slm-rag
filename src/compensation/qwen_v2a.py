"""XGrammar view over the already loaded Qwen; no second model allocation."""
from copy import copy
import importlib.metadata

from src.compensation.qwen_constrained import QwenConstrainedGenerator, XGRAMMAR_VERSION


class QwenV2AGenerator(QwenConstrainedGenerator):
    @classmethod
    def from_shared(cls, generator):
        if importlib.metadata.version('xgrammar') != XGRAMMAR_VERSION:
            raise RuntimeError('E v2a requires xgrammar==' + XGRAMMAR_VERSION)
        if getattr(generator.model.generation_config, 'prefill_chunk_size', None) != 256:
            raise RuntimeError('E v2a must share the registered prefill256 runtime')
        import xgrammar as xgr
        from xgrammar.contrib.hf import LogitsProcessor
        view = cls.__new__(cls)
        view.__dict__.update(generator.__dict__)
        view.runtime_environment = copy(generator.runtime_environment)
        view.json_output = True
        view.xgr, view.processor_class = xgr, LogitsProcessor
        info = xgr.TokenizerInfo.from_huggingface(view.tokenizer, vocab_size=view.model.config.vocab_size)
        view.compiler = xgr.GrammarCompiler(info, max_threads=4)
        view.runtime_environment.update(xgrammar=XGRAMMAR_VERSION, structured_decoder='xgrammar_hf_logits_processor',
                                        v2a_shared_model=True, prefill_chunk_size=256)
        return view

    def generate_schema(self, prompt, max_new_tokens, schema):
        if getattr(self.model.generation_config, 'prefill_chunk_size', None) != 256:
            raise RuntimeError('Registered chunked prefill was changed')
        generated, meta = super().generate_schema(prompt, max_new_tokens, schema)
        return generated, {**meta, 'prefill_chunk_size': 256, 'shared_model': True}
