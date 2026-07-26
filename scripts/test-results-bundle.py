#!/usr/bin/env python3
"""Focused regression test for historical benchmark metadata."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_results_bundle", ROOT / "scripts" / "build-results-bundle.py",
)
assert SPEC and SPEC.loader
bundle = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bundle)


def main() -> int:
    data = {
        "profile": "profile-v1",
        "framework": "llama-benchy",
        "measurements": [{"depth": 0, "pp": 512, "tg": 128, "concurrency": 1}],
        "failed_points": 0,
        "skipped_points": 0,
        "max_context": 4096,
        "meta": {
            "benchmark_spec": {
                "file": "benchmarking/profile-v1.yaml",
                "framework": "recorded-client",
                "metadata": {"version": "1"},
                "args": {"prefix_caching": True},
            },
        },
    }
    live_profile = {
        "framework": "changed-client",
        "metadata": {"version": "2"},
        "args": {"prefix_caching": False},
    }
    with mock.patch.object(bundle, "profile_path", return_value=Path("profile-v1.yaml")), \
            mock.patch.object(bundle, "read_yaml", return_value=live_profile):
        params = bundle.benchmark_params(data)

    assert params["framework"] == "recorded-client"
    assert params["metadata"] == {"version": "1"}
    assert params["args"] == {"prefix_caching": True}
    assert params["profile_file"] == "benchmarking/profile-v1.yaml"
    assert params["measurement_count"] == 1
    print("historical benchmark metadata remains immutable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
