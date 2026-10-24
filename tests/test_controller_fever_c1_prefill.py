"""Registered chunked prefill retains complete input and greedy generation."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from scripts.run_controller_fever_c1 import (PREFILL_RUNTIME, configure_prefill,
                                              generation_runtime, engineering_smoke)


class RuntimeTests(unittest.TestCase):
    def test_long_engineering_fixture_uses_synthetic_text_and_covers_registered_shape(self):
        from src.compensation.answer_repair import Generation
        class Tokenizer:
            def encode(self,text,**kwargs):return text.split()
            def apply_chat_template(self,messages,**kwargs):return messages[0]['content']
        class Generator:
            tokenizer=Tokenizer()
            def generate(self,prompt,budget):
                self_outer.assertIn('fictional',prompt)
                self_outer.assertEqual(budget,64)
                return Generation('SUPPORTS',.01,len(prompt.split()),1)
        self_outer=self
        smoke=engineering_smoke(Generator(),min_input_tokens=200)
        self.assertTrue(smoke['passed'])
        self.assertEqual(smoke['llm_calls'],3)
        self.assertGreaterEqual(smoke['traces'][-1]['generation']['input_tokens'],200)

    def test_legacy_defaults_unchanged_and_unknown_runtime_rejected(self):
        self.assertIsNone(generation_runtime({}))
        self.assertEqual(generation_runtime({'generation_runtime':PREFILL_RUNTIME}), PREFILL_RUNTIME)
        for value in ({'prefill_chunk_size':0},
                      {**PREFILL_RUNTIME,'prefill_chunk_size':True},
                      {**PREFILL_RUNTIME,'truncate_evidence':True}):
            with self.assertRaises(ValueError):generation_runtime({'generation_runtime':value})

    def test_native_configuration_and_environment_recorded_without_model_change(self):
        from transformers import GenerationConfig
        config=GenerationConfig()
        model=SimpleNamespace(generation_config=config)
        generator=SimpleNamespace(model=model,runtime_environment={})
        configure_prefill(generator,None)
        self.assertIsNone(config.prefill_chunk_size)
        configure_prefill(generator,PREFILL_RUNTIME)
        self.assertIs(generator.model,model)
        self.assertEqual(config.prefill_chunk_size,256)
        self.assertEqual(generator.runtime_environment['generation_runtime'],PREFILL_RUNTIME)
        self.assertEqual(generator.runtime_environment['generation_config']['prefill_chunk_size'],256)
        with self.assertRaises(RuntimeError):
            configure_prefill(SimpleNamespace(model=SimpleNamespace(generation_config=object())),PREFILL_RUNTIME)

    def test_real_transformers_qwen2_chunking_bounds_prefill_and_preserves_full_sequence(self):
        import subprocess, sys
        code = """\
from unittest.mock import patch
import torch
import torch.nn.functional as functional
from transformers import Qwen2Config, Qwen2ForCausalLM
torch.set_num_threads(1)
torch.manual_seed(42)
model=Qwen2ForCausalLM(Qwen2Config(vocab_size=64,hidden_size=32,intermediate_size=64,
    num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,
    max_position_embeddings=256,eos_token_id=None,pad_token_id=0)).eval()
inputs=torch.arange(1,65).remainder(63).add(1).reshape(1,-1)
mask=torch.ones_like(inputs)
with torch.inference_mode():
    original=model.generate(input_ids=inputs,attention_mask=mask,max_new_tokens=4,do_sample=False)
calls=[]
sdpa=functional.scaled_dot_product_attention
def observe(query,key,value,*args,**kwargs):
    calls.append((query.shape[-2],key.shape[-2]))
    return sdpa(query,key,value,*args,**kwargs)
model.generation_config.prefill_chunk_size=8
with torch.inference_mode(), patch.object(functional,'scaled_dot_product_attention',side_effect=observe):
    chunked=model.generate(input_ids=inputs,attention_mask=mask,max_new_tokens=4,do_sample=False)
assert torch.equal(original,chunked)
assert torch.equal(chunked[:,:64],inputs)
assert chunked.shape[-1] == 68
assert max(q for q,k in calls) == 8
assert (8,64) in calls  # Last prefill chunk still attends to all 64 tokens.
"""
        result=subprocess.run([sys.executable,"-c",code],capture_output=True,text=True,timeout=180)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)


if __name__=='__main__':
    unittest.main()
