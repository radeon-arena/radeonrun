#!/usr/bin/env python3
"""Build browser-friendly result manifests from committed radeonrun results.

Outputs:
  results/index.json   small file list + repo metadata
  results/bundle.json  all result JSON documents in one fetchable payload

The Radeon Arena static site reads bundle.json directly from GitHub raw.
"""
from __future__ import annotations

import datetime as dt
import json
import subprocess
from pathlib import Path

try:
    import yaml
except ImportError as exc:  # pragma: no cover - developer/runtime dependency check
    raise SystemExit("PyYAML is required: pip install pyyaml") from exc

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DEVICES = ("strix", "w7900", "r9700")


def git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except Exception:
        return ""


def main() -> None:
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
            recipe_path = ROOT / "recipes" / f"{stem}.yaml"
            data = json.loads(path.read_text())
            recipe = yaml.safe_load(recipe_path.read_text()) if recipe_path.exists() else None
            records[device].append({
                "file": f"results/{device}/{name}",
                "recipe_file": f"recipes/{stem}.yaml" if recipe is not None else None,
                "recipe": recipe,
                "data": data,
            })

    index = {
        "version": 1,
        "generated_at": generated_at,
        "repo": "radeon-arena/radeonrun",
        "commit": commit,
        "short_commit": short,
        "devices": devices,
    }
    bundle = {**index, "records": records}

    (RESULTS / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n")
    (RESULTS / "bundle.json").write_text(json.dumps(bundle, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote results/index.json and results/bundle.json ({sum(len(v) for v in records.values())} records)")


if __name__ == "__main__":
    main()
