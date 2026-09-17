"""Load the frozen project config from configs/base.yaml."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "base.yaml"


def load_base_config(path: Path | None = None) -> dict:
    config_path = path or CONFIG_PATH
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise RuntimeError(f"Invalid config at {config_path}")
    return config


def load_generator_config(path: Path | None = None) -> dict:
    generator = load_base_config(path).get("generator")
    if not isinstance(generator, dict) or not generator.get("model_name"):
        raise RuntimeError(
            "configs/base.yaml must define generator.model_name; "
            "refusing to fall back to a hardcoded model."
        )
    return generator
