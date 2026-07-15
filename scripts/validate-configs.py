#!/usr/bin/env python3
"""Validate composable RadeonRun configs, legacy compatibility and results."""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from radeonrun_config import (  # noqa: E402
    ConfigError,
    list_run_configs,
    load_run_config,
    render_command,
    resolve_image,
)

DEVICES = {"halo": "strix", "w7900": "w7900", "r9700": "r9700"}
REQUIRED_AXES = ("device", "model", "launch", "benchmark")


def error(message: str, errors: list[str]) -> None:
    print(message)
    errors.append(message)


def normalized_command(config: dict) -> str:
    """Match the shell command normalization performed by run-recipe.py."""
    return " ".join(render_command(config).replace("\\\n", " ").split())


def validate_configs(errors: list[str]) -> None:
    names = list_run_configs()
    if not names:
        error("no run configurations found", errors)
        return
    for name in names:
        try:
            config = load_run_config(name)
        except ConfigError as exc:
            error(f"{name}: {exc}", errors)
            continue
        for key in ("name", "model", "source", "container", "command"):
            if not config.get(key):
                error(f"{name}: normalized config missing {key}", errors)
        if config.get("_config_source") == "matrix":
            for axis in ("_device", "_model", "_launch", "_benchmark"):
                if not isinstance(config.get(axis), dict):
                    error(f"{name}: matrix missing resolved {axis}", errors)
        try:
            device_id = str((config.get("_device") or {}).get("id") or "halo")
            default_image = resolve_image(config, device_id)
            if not default_image:
                error(f"{name}: no resolved image", errors)
        except ConfigError as exc:
            error(f"{name}: image resolution failed: {exc}", errors)
            default_image = ""
            device_id = ""

        launch = config.get("_launch") or {}
        topology = launch.get("topology") or {}
        gpu_count = int(topology.get("gpu_count") or 1)
        tp = int(topology.get("tensor_parallel_size") or 1)
        node_count = int(topology.get("node_count") or 1)
        env = config.get("env") or {}
        if min(gpu_count, tp, node_count) < 1:
            error(f"{name}: topology counts must be positive", errors)
        if gpu_count != tp:
            error(f"{name}: topology gpu_count={gpu_count} must equal tensor_parallel_size={tp}", errors)
        if node_count != 1:
            error(f"{name}: only single-node launches are currently supported", errors)
        if tp > 1:
            if env.get("NCCL_PROTO") != "Simple":
                error(f"{name}: TP>1 requires NCCL_PROTO=Simple", errors)
            if str(env.get("NCCL_P2P_DISABLE")) != "1":
                error(f"{name}: TP>1 requires NCCL_P2P_DISABLE=1", errors)
            if f"--tensor-parallel-size {tp}" not in normalized_command(config):
                error(f"{name}: topology TP{tp} does not match launch command", errors)
        visible = str(env.get("HIP_VISIBLE_DEVICES") or "")
        if visible:
            visible_count = len([v for v in visible.split(",") if v.strip()])
            if visible_count != gpu_count:
                error(f"{name}: HIP_VISIBLE_DEVICES exposes {visible_count} GPUs, topology declares {gpu_count}", errors)
        if device_id == "r9700":
            if "@sha256:" not in default_image:
                error(f"{name}: R9700 image must be pinned by digest", errors)
            if env.get("HIP_VISIBLE_DEVICES") != env.get("ROCR_VISIBLE_DEVICES"):
                error(f"{name}: R9700 HIP/ROCR visible device sets must match", errors)

    # Explicit OCI refs are provider-owned and must never be rewritten.
    sample = load_run_config(names[0])
    external = "quay.io/example/custom-vllm@sha256:deadbeef"
    if resolve_image(sample, "w7900", image_override=external) != external:
        error("explicit OCI image was rewritten", errors)


def validate_legacy_parity(errors: list[str]) -> None:
    # During migration every legacy recipe is shadowed by a matrix.  Flat fields
    # remain byte-compatible for external consumers while execution uses specs.
    for path_str in sorted(glob.glob(str(ROOT / "recipes" / "*.yaml"))):
        path = Path(path_str)
        name = path.stem
        legacy = yaml.safe_load(path.read_text()) or {}
        normalized = load_run_config(name)
        if normalized.get("_config_source") != "matrix":
            error(f"{name}: legacy recipe is not represented in matrices", errors)
        for key in ("model", "source", "container"):
            if normalized.get(key) != legacy.get(key):
                error(f"{name}: matrix changes legacy {key}", errors)
        for key in ("defaults", "env"):
            if (normalized.get(key) or {}) != (legacy.get(key) or {}):
                error(f"{name}: matrix changes legacy {key}", errors)
        if normalized_command(normalized) != normalized_command(legacy):
            error(f"{name}: matrix changes rendered command", errors)


def validate_results(errors: list[str]) -> None:
    for path_str in sorted(glob.glob(str(ROOT / "results" / "*" / "*.json"))):
        path = Path(path_str)
        if path.parent.name not in DEVICES.values():
            continue
        if path.stem not in list_run_configs():
            error(f"{path.relative_to(ROOT)}: missing run configuration", errors)
        data = json.loads(path.read_text())
        if not data.get("measurements"):
            error(f"{path.relative_to(ROOT)}: no measurements", errors)
        for measurement in data.get("measurements", []):
            if ("decode_toks_per_s" not in measurement and "prefill_toks_per_s" not in measurement) or "concurrency" not in measurement:
                error(f"{path.relative_to(ROOT)}: incomplete measurement {measurement}", errors)
            if data.get("profile") == "halo-arena-v2" and "depth" not in measurement:
                error(f"{path.relative_to(ROOT)}: v2 measurement missing depth", errors)


def validate_bundle(errors: list[str]) -> None:
    index = json.loads((ROOT / "results" / "index.json").read_text())
    bundle = json.loads((ROOT / "results" / "bundle.json").read_text())
    if int(bundle.get("version") or 0) < 2:
        error("bundle version must be >= 2", errors)
    for result_key in DEVICES.values():
        files = sorted(os.path.basename(p) for p in glob.glob(str(ROOT / "results" / result_key / "*.json")))
        if index.get("devices", {}).get(result_key, []) != files:
            error(f"index mismatch for {result_key}", errors)
        records = bundle.get("records", {}).get(result_key, [])
        bundled = sorted(os.path.basename(record["file"]) for record in records)
        if bundled != files:
            error(f"bundle mismatch for {result_key}", errors)
        for record in records:
            for axis in REQUIRED_AXES:
                if not isinstance(record.get(axis), dict):
                    error(f"{record.get('file')}: missing structured {axis}", errors)
            if not record.get("matrix_file") or not record.get("model_file") or not record.get("launch_file"):
                error(f"{record.get('file')}: missing spec provenance", errors)
            launch = record.get("launch") or {}
            if not launch.get("image"):
                error(f"{record.get('file')}: launch.image missing", errors)
            topology = launch.get("topology") or {}
            if topology and int(topology.get("gpu_count") or 1) != int(topology.get("tensor_parallel_size") or 1):
                error(f"{record.get('file')}: launch topology GPU/TP mismatch", errors)


def main() -> int:
    errors: list[str] = []
    validate_configs(errors)
    validate_legacy_parity(errors)
    validate_results(errors)
    validate_bundle(errors)
    if errors:
        print(f"validation failed: {len(errors)} error(s)")
        return 1
    print(f"validation ok: {len(list_run_configs())} configs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
