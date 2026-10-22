"""Lazy XGrammar backend over the project's exact shared Qwen NF4 model."""
from __future__ import annotations

import importlib.metadata
import json
import time

from src.compensation.answer_repair import Generation
from src.compensation.qwen_repair import QwenRepairGenerator

XGRAMMAR_VERSION = "0.2.8"


class QwenConstrainedGenerator(QwenRepairGenerator):
    def __init__(self):
        if importlib.metadata.version("xgrammar") != XGRAMMAR_VERSION:
            raise RuntimeError("Contract experiment requires xgrammar==" + XGRAMMAR_VERSION)
        import xgrammar as xgr
        from xgrammar.contrib.hf import LogitsProcessor
        super().__init__(json_output=True)
        self.xgr, self.processor_class = xgr, LogitsProcessor
        info = xgr.TokenizerInfo.from_huggingface(self.tokenizer, vocab_size=self.model.config.vocab_size)
        self.compiler = xgr.GrammarCompiler(info, max_threads=4)
        self.runtime_environment.update(xgrammar=XGRAMMAR_VERSION, structured_decoder="xgrammar_hf_logits_processor")

    def generate_schema(self, prompt, max_new_tokens, schema):
        import torch
        from src.generation.qwen import apply_qwen_chat_template, _input_device
        started = time.perf_counter()
        try:
            grammar = self.compiler.compile_json_schema(json.dumps(schema, ensure_ascii=False))
        except Exception as exc:
            raise RuntimeError("XGrammar cannot compile this frozen schema; refusing silent fallback") from exc
        compile_seconds = time.perf_counter() - started
        processor = self.processor_class(grammar)  # fresh matcher for every call
        chat = apply_qwen_chat_template(self.tokenizer, prompt)
        inputs = self.tokenizer(chat, return_tensors="pt")
        input_tokens = inputs["input_ids"].shape[1]
        limit = getattr(self.model.config, "max_position_embeddings", None)
        if limit is not None and input_tokens + max_new_tokens > limit:
            raise RuntimeError("Contract prompt exceeds context; refusing truncation")
        inputs = {key: value.to(_input_device(self.model)) for key, value in inputs.items()}
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                         pad_token_id=self.tokenizer.pad_token_id, logits_processor=[processor])
        torch.cuda.synchronize()
        latency = time.perf_counter() - started
        tokens = output[0, input_tokens:]
        raw = self.tokenizer.decode(tokens, skip_special_tokens=True).strip()
        generation = Generation(raw, latency, input_tokens,
                                len(self.tokenizer.encode(raw, add_special_tokens=False)), raw_text=raw)
        return generation, {"decoder": "xgrammar", "version": XGRAMMAR_VERSION,
                            "schema_compile_seconds": compile_seconds,
                            "output_tokens_including_stop": len(tokens),
                            "hit_token_budget": len(tokens) >= max_new_tokens}

    def generate_unconstrained_schema(self, prompt, max_new_tokens, schema):
        return super().generate(prompt, max_new_tokens), {"decoder": "prompt_only", "schema_compile_seconds": 0.0}
