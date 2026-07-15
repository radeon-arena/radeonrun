#!/usr/bin/env python3
"""Build browser-friendly result manifests from committed radeonrun results.

Outputs:
  results/index.json   small file list + repo metadata
  results/bundle.json  all result JSON documents in one fetchable payload

The Radeon Arena static site reads bundle.json directly from GitHub raw.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except ImportError as exc:  # pragma: no cover - developer/runtime dependency check
    raise SystemExit("PyYAML is required: pip install pyyaml") from exc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from radeonrun_config import (  # noqa: E402
    ConfigError,
    load_run_config,
    public_recipe,
    resolve_image,
    resolved_axes,
)

RESULTS = ROOT / "results"
DEVICES = ("strix", "w7900", "r9700")
RESULT_TO_DEVICE = {"strix": "halo", "w7900": "w7900", "r9700": "r9700"}

def git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except Exception:
        return ""


def read_yaml(path: Path) -> dict | None:
    if not path.exists():
        return None
    return yaml.safe_load(path.read_text()) or {}


def profile_path(profile: str | None) -> Path | None:
    if not profile:
        return None
    name = profile if profile.endswith(".yaml") else f"{profile}.yaml"
    path = ROOT / "benchmarking" / name
    return path if path.exists() else None


def benchmark_params(data: dict) -> dict:
    profile = data.get("profile")
    path = profile_path(profile)
    profile_doc = read_yaml(path) if path else None
    measurements = data.get("measurements") or []
    point_params = []
    seen = set()
    for measurement in measurements:
        key = (
            measurement.get("depth", 0),
            measurement.get("pp"),
            measurement.get("tg"),
            measurement.get("concurrency", 1),
        )
        if key in seen:
            continue
        seen.add(key)
        point_params.append({
            "depth": key[0],
            "pp": key[1],
            "tg": key[2],
            "concurrency": key[3],
        })
    return {
        "profile": profile,
        "profile_file": f"benchmarking/{path.name}" if path else None,
        "framework": data.get("framework") or (profile_doc or {}).get("framework"),
        "metadata": (profile_doc or {}).get("metadata") or {},
        "args": (profile_doc or {}).get("args") or {},
        "schedule": (profile_doc or {}).get("schedule") or None,
        "measurement_count": len(measurements),
        "point_params": point_params,
        "failed_points": data.get("failed_points"),
        "skipped_points": data.get("skipped_points"),
        "max_context": data.get("max_context"),
    }


def structured_record(device: str, name: str, config: dict, data: dict) -> dict:
    meta = data.get("meta") or {}
    actual_image = meta.get("image_requested") or meta.get("image")
    if not actual_image:
        actual_image = resolve_image(config, RESULT_TO_DEVICE.get(device, device))
    axes = resolved_axes(config, str(actual_image))
    axes["device"]["result_key"] = device

    model_meta = meta.get("model_spec") if isinstance(meta.get("model_spec"), dict) else {}
    launch_meta = meta.get("launch_spec") if isinstance(meta.get("launch_spec"), dict) else {}
    benchmark_meta = meta.get("benchmark_spec") if isinstance(meta.get("benchmark_spec"), dict) else {}
    axes["model"].update(model_meta)
    axes["launch"].update(launch_meta)
    axes["benchmark"].update(benchmark_meta)
    for key in ("image", "image_requested", "image_resolved", "image_digest", "image_tag", "image_commit", "image_id"):
        if meta.get(key) is not None:
            axes["launch"][key] = meta[key]
    axes["launch"]["command"] = meta.get("command") or config.get("command")
    axes["benchmark"].update(benchmark_params(data))
    return axes


def _stable_payload(payload: dict) -> dict:
    stable = dict(payload)
    for key in ("generated_at", "commit", "short_commit"):
        stable.pop(key, None)
    return stable


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or verify RadeonRun result manifests")
    parser.add_argument("--check", action="store_true", help="Verify committed manifests, ignoring volatile timestamp/commit metadata")
    args = parser.parse_args()
    generated_at = dt.datetime.now(dt.timezone.utc).isoformat()
    commit = git("rev-parse", "HEAD")
    short = git("rev-parse", "--short", "HEAD")
    devices: dict[str, list[str]] = {}
    records: dict[str, list[dict]] = {}

    for device in DEVICES:
        d = RESULTS / device
        files = sorted(p.name for p in d.glob("*.json")) if d.exists() else []
        devices[device] = files
        records[device] = []
        for name in files:
            path = d / name
            stem = path.stem
            data = json.loads(path.read_text())
            result_device = RESULT_TO_DEVICE[device]
            try:
                config = load_run_config(
                    stem,
                    device_override=result_device,
                    benchmark_override=data.get("profile"),
                )
            except ConfigError as exc:
                raise SystemExit(f"{path}: {exc}") from exc
            spec_files = config.get("_spec_files") or {}
            recipe = public_recipe(config)
            records[device].append({
                "file": f"results/{device}/{name}",
                "config_source": config.get("_config_source", "legacy"),
                "spec_files": spec_files,
                "matrix_file": spec_files.get("matrix"),
                "model_file": spec_files.get("model"),
                "launch_file": spec_files.get("launch"),
                "device_file": spec_files.get("device"),
                "benchmark_file": spec_files.get("benchmark"),
                "recipe_file": spec_files.get("legacy_recipe"),
                "recipe": recipe,
                "data": data,
                **structured_record(device, name, config, data),
            })

    index = {
        "version": 2,
        "generated_at": generated_at,
        "repo": "radeon-arena/radeonrun",
        "commit": commit,
        "short_commit": short,
        "devices": devices,
    }
    bundle = {**index, "records": records}

    index_path = RESULTS / "index.json"
    bundle_path = RESULTS / "bundle.json"
    if args.check:
        committed_index = json.loads(index_path.read_text())
        committed_bundle = json.loads(bundle_path.read_text())
        if _stable_payload(committed_index) != _stable_payload(index):
            raise SystemExit("results/index.json is stale; run scripts/build-results-bundle.py")
        if _stable_payload(committed_bundle) != _stable_payload(bundle):
            raise SystemExit("results/bundle.json is stale; run scripts/build-results-bundle.py")
        print(f"result manifests are current ({sum(len(v) for v in records.values())} records)")
        return

    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n")
    bundle_path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote results/index.json and results/bundle.json ({sum(len(v) for v in records.values())} records)")


if __name__ == "__main__":
    main()
