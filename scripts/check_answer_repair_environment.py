"""Check the E GPU environment without loading or downloading a model."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path


def inspect_environment() -> dict:
    packages = {}
    for name in ("torch", "transformers", "accelerate", "bitsandbytes", "PyYAML"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": packages, "cuda_available": False, "cuda_version": None,
        "gpus": [], "errors": [],
    }
    if sys.version_info < (3, 11):
        report["errors"].append("Python 3.11 or newer is required")
    for name, version in packages.items():
        if version is None:
            report["errors"].append(f"Missing package: {name}")
    if packages["torch"] is not None:
        try:
            import torch
            report["cuda_available"] = torch.cuda.is_available()
            report["cuda_version"] = torch.version.cuda
            if report["cuda_available"]:
                report["gpus"] = [
                    {"index": i, "name": torch.cuda.get_device_name(i),
                     "total_memory_bytes": torch.cuda.get_device_properties(i).total_memory}
                    for i in range(torch.cuda.device_count())
                ]
        except Exception as exc:
            report["errors"].append(f"PyTorch initialization failed: {type(exc).__name__}: {exc}")
    report["ready_for_nf4_smoke"] = report["cuda_available"] and not report["errors"]
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="Save the environment report")
    parser.add_argument("--require-cuda", action="store_true", help="Fail if prerequisites are missing")
    args = parser.parse_args(argv)
    report = inspect_environment()
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return int(args.require_cuda and not report["ready_for_nf4_smoke"])


if __name__ == "__main__":
    raise SystemExit(main())
