#!/usr/bin/env python3
"""Focused regression tests for --visible-devices GPU pinning."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SPEC = importlib.util.spec_from_file_location("run_recipe", ROOT / "run-recipe.py")
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def recipe(gpu_count: int, tp: int) -> dict:
    """A resolved recipe shaped like the ones run-recipe.py builds."""
    return {
        "env": {"HIP_VISIBLE_DEVICES": "0", "ROCR_VISIBLE_DEVICES": "0"},
        "_launch": {
            "topology": {"node_count": 1, "gpu_count": gpu_count, "tensor_parallel_size": tp},
            "env": {"HIP_VISIBLE_DEVICES": "0", "ROCR_VISIBLE_DEVICES": "0"},
        },
    }


def rejects(spec: str, gpu_count: int) -> str:
    try:
        runner._apply_visible_devices(recipe(gpu_count, gpu_count), spec)
    except ValueError as exc:
        return str(exc)
    raise AssertionError(f"expected {spec!r} to be rejected for gpu_count={gpu_count}")


def main() -> int:
    # A vLLM TP2 run pinned away from a busy GPU rewrites both env scopes.
    tp2 = recipe(2, 2)
    runner._apply_visible_devices(tp2, "2,3")
    for scope in (tp2, tp2["_launch"]):
        assert scope["env"]["HIP_VISIBLE_DEVICES"] == "2,3", scope
        assert scope["env"]["ROCR_VISIBLE_DEVICES"] == "2,3", scope

    # llama.cpp splits layers instead of sharding, so gpu_count may exceed tp.
    multi = recipe(3, 1)
    runner._apply_visible_devices(multi, " 0 , 2 , 3 ")
    assert multi["env"]["HIP_VISIBLE_DEVICES"] == "0,2,3", multi

    # The device count must match what the launch topology declares.
    assert "declares 1" in rejects("0,2", 1)
    assert "declares 2" in rejects("3", 2)
    assert "no GPU indices given" in rejects(" , ", 1)

    print("visible device pinning ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
