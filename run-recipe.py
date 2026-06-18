#!/usr/bin/env python3
"""
run-recipe.py - One-click model deployment using YAML recipes (SCAFFOLD)

Mirrors the upstream interface so recipes can be filled in later. This is a
skeleton: argument parsing and the recipe schema are sketched, but the actual
build/download/launch orchestration for ROCm/Strix Halo is NOT implemented.

Intended responsibilities (TODO):
- Load a recipe YAML from recipes/<name>.yaml
- Optionally download the model from HuggingFace (--setup / --download-model)
- Optionally build the ROCm container (build-and-copy.sh)
- Apply mods listed in the recipe
- Generate and run the launch command (solo or cluster)
"""

import argparse
import sys
from pathlib import Path

RECIPES_DIR = Path(__file__).resolve().parent / "recipes"


def list_recipes() -> None:
    if not RECIPES_DIR.is_dir():
        print("No recipes/ directory found.")
        return
    found = sorted(p.stem for p in RECIPES_DIR.glob("*.yaml"))
    if not found:
        print("No recipes found (scaffold).")
        return
    print("Available recipes:")
    for name in found:
        print(f"  - {name}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="One-click vLLM recipe runner for Strix Halo (scaffold).",
    )
    parser.add_argument("recipe", nargs="?", help="Recipe name (without .yaml)")
    parser.add_argument("--list", action="store_true", help="List available recipes")
    parser.add_argument("--solo", action="store_true", help="Single-node mode")
    parser.add_argument("--setup", action="store_true", help="Build + download before run")
    parser.add_argument("--download-model", action="store_true", help="Download the model only")
    parser.add_argument("-n", "--nodes", help="Comma-separated cluster node IPs")
    parser.add_argument("--port", type=int, help="Override serve port")
    parser.add_argument("--gpu-mem", type=float, help="Override gpu_memory_utilization")
    args = parser.parse_args()

    if args.list or not args.recipe:
        list_recipes()
        return 0

    # TODO: implement recipe loading + orchestration for ROCm/Strix Halo.
    print(f"[scaffold] recipe '{args.recipe}' selected, but the runner is not "
          f"implemented yet. See run-recipe.py TODOs.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
