#!/usr/bin/env python3
"""
run-recipe.py - One-click recipe runner.

Loads a recipe YAML, fills the command template with its defaults (plus any CLI
overrides), and runs it via launch-cluster.sh --solo (or prints it with
--print). The serve commands in the recipes are the ones run on the
InferStation gfx1151 fleet.

Examples:
  ./run-recipe.py --list
  ./run-recipe.py qwen3.6-35b-a3b-bf16 --print
  MODELS_DIR=/models ./run-recipe.py qwen3.6-35b-a3b-bf16
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
        print("No recipes found.")
        return
    print("Available recipes:")
    for name in found:
        print(f"  - {name}")


def _render_command(recipe: dict, overrides: dict) -> str:
    """Fill the recipe's command template with defaults + CLI overrides."""
    import yaml  # noqa: F401  (already importable; used by callers)
    params = dict(recipe.get("defaults") or {})
    for k, v in overrides.items():
        if v is not None:
            params[k] = v
    cmd = (recipe.get("command") or "").strip()
    for key, val in params.items():
        cmd = cmd.replace("{" + key + "}", str(val))
    return cmd


def main() -> int:
    import yaml

    parser = argparse.ArgumentParser(
        description="One-click recipe runner for Strix Halo (gfx1151).",
    )
    parser.add_argument("recipe", nargs="?", help="Recipe name (without .yaml)")
    parser.add_argument("--list", action="store_true", help="List available recipes")
    parser.add_argument("--solo", action="store_true", help="Single-node mode (default)")
    parser.add_argument("--print", dest="print_only", action="store_true",
                        help="Print the launch command instead of running it")
    parser.add_argument("--port", type=int, help="Override serve port")
    parser.add_argument("--nseq", type=int, help="Override --max-num-seqs / -np")
    parser.add_argument("--ctx", type=int, help="Override llama.cpp -c context")
    args = parser.parse_args()

    if args.list or not args.recipe:
        list_recipes()
        return 0

    path = RECIPES_DIR / f"{args.recipe}.yaml"
    if not path.is_file():
        print(f"Recipe not found: {path}")
        return 2

    recipe = yaml.safe_load(path.read_text())
    cmd = _render_command(recipe, {"port": args.port, "nseq": args.nseq, "ctx": args.ctx})
    if not cmd:
        print(f"Recipe '{args.recipe}' has no command.")
        return 2

    container = recipe.get("container", "halo-vllm-node")
    # Build the launch-cluster.sh invocation. The recipe command already has
    # the model path baked in; we just wrap it in the solo launcher.
    launch = (RECIPES_DIR.parent / "launch-cluster.sh")
    port = args.port or (recipe.get("defaults") or {}).get("port", 8000)
    inner = cmd.replace("\\\n", " ")        # drop line-continuation backslashes
    inner = " ".join(inner.split())          # collapse whitespace
    full = f'IMAGE={container} {launch} --solo -p {port}:{port} exec {inner}'

    print(full)
    if args.print_only:
        return 0

    import subprocess
    return subprocess.call(full, shell=True)


if __name__ == "__main__":
    sys.exit(main())
