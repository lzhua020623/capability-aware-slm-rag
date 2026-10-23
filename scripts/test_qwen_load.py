"""Minimal 4-bit NF4 load test for the configured Qwen generator.

Does not call the retriever, load datasets, or write result files.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_generator_config
from src.generation.qwen import cpu_offload_enabled, generate_greedy, load_qwen_nf4


def main() -> int:
    generator = load_generator_config()
    model_name = str(generator["model_name"])
    print(f"Loading {model_name} with 4-bit NF4 from configs/base.yaml")
    tokenizer, model = load_qwen_nf4()
    allocated_mib = (
        torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
    )
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    answer, _latency = generate_greedy(
        tokenizer,
        model,
        "Reply with the single word: ok",
        max_new_tokens=8,
    )
    print(f"loaded model name: {model_name}")
    print("quantization = 4-bit NF4")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU name: {gpu_name}")
    print(f"GPU allocated memory: {allocated_mib:.1f} MiB")
    print(f"CPU offload enabled: {cpu_offload_enabled()}")
    print(f"device map: {getattr(model, 'hf_device_map', None)}")
    if hasattr(model, "get_memory_footprint"):
        print(f"model memory footprint: {model.get_memory_footprint() / (1024**3):.2f} GiB")
    print(f"simple generation result: {answer}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
