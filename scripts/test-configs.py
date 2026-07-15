#!/usr/bin/env python3
"""Focused regression tests for composable config resolution."""
from __future__ import annotations

import tempfile
from pathlib import Path

import yaml

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import radeonrun_config as config  # noqa: E402


def dump(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="radeonrun-config-") as temp:
        root = Path(temp)
        old = {
            name: getattr(config, name)
            for name in ("ROOT", "RECIPES_DIR", "MATRICES_DIR", "MODELS_DIR", "LAUNCHES_DIR", "DEVICES_DIR", "BENCHMARKING_DIR")
        }
        try:
            config.ROOT = root
            config.RECIPES_DIR = root / "recipes"
            config.MATRICES_DIR = root / "matrices"
            config.MODELS_DIR = root / "models"
            config.LAUNCHES_DIR = root / "launches"
            config.DEVICES_DIR = root / "devices"
            config.BENCHMARKING_DIR = root / "benchmarking"

            dump(config.MODELS_DIR / "model-a.yaml", {
                "id": "model-a", "path": "/models/a", "source": "org/a",
                "served_name": "a", "quantization": "BF16",
            })
            dump(config.LAUNCHES_DIR / "launch-a.yaml", {
                "id": "launch-a", "runtime": "vllm", "container": "vllm",
                "image": {"ref": "quay.io/acme/vllm@sha256:deadbeef"},
                "defaults": {"host": "0.0.0.0", "port": 8000, "ctx": 4096, "nseq": 2},
                "command": "vllm serve {model.path} --served-model-name {model.served_name} --host {host} --port {port} --max-model-len {ctx} --max-num-seqs {nseq}",
            })
            for device, arch in (("halo", "gfx1151"), ("w7900", "gfx1100")):
                dump(config.DEVICES_DIR / f"{device}.yaml", {
                    "id": device, "label": device, "gpu": device, "arch": arch,
                    "image_device": device, "result_key": "strix" if device == "halo" else device,
                })
            dump(config.BENCHMARKING_DIR / "profile-a.yaml", {
                "framework": "llama-benchy", "args": {"pp": [512], "tg": [128], "concurrency": [1]},
            })
            dump(config.MATRICES_DIR / "matrix-a.yaml", {
                "id": "matrix-a", "model": "model-a", "launch": "launch-a",
                "device": "halo", "benchmark": "profile-a",
            })

            resolved = config.load_run_config("matrix-a")
            assert resolved["_config_source"] == "matrix"
            assert resolved["_spec_files"]["legacy_recipe"] is None
            assert resolved["model"] == "/models/a"
            assert config.resolve_image(resolved, "halo") == "quay.io/acme/vllm@sha256:deadbeef"
            assert config.resolve_image(resolved, "w7900") == "quay.io/acme/vllm@sha256:deadbeef"
            assert config.render_command(resolved, {"port": 9000}).endswith("--max-num-seqs 2")
            assert "--port 9000" in config.render_command(resolved, {"port": 9000})

            overridden = config.load_run_config("matrix-a", device_override="w7900")
            assert overridden["_device"]["arch"] == "gfx1100"
            assert config.list_run_configs() == ["matrix-a"]
        finally:
            for name, value in old.items():
                setattr(config, name, value)
    print("standalone matrix resolution ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
