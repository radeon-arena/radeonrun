# Recipes

Recipes are a **one-click** way to deploy a model with pre-configured settings.
Each recipe is a YAML file specifying:

- the model (local `/models/...` path or HF id)
- container image and build arguments
- required mods/patches
- default parameters (port, host, concurrency, etc.)
- environment variables
- the serve command (vLLM or llama-server)

The serve commands in these recipes are **the ones run on the InferStation
gfx1151 fleet** (halo5 / halo6) and produce its daily benchmark results. Run one
with `./run-recipe.py <name>` (or `--print` to show the command), or paste its
serve command into `launch-cluster.sh --solo` directly.

## Verified recipes

All recipes here were generated from the InferStation gfx1151 unit definitions
and cross-checked against `runs.json` — each one corresponds to a config that
produced a real decode result on halo5 / halo6 (the comment in each file records
the best measured tok/s). FLASH_ATTN configs (which fail on gfx1151) are
excluded.

**37 recipes** across two engines:

- **vLLM** (`container: halo-vllm-node`, 9): Qwen3.6-35B-A3B (BF16 / AWQ-4bit /
  Quark-W8A8), Qwen3.6-27B (BF16 / Quark), Qwen3-30B-A3B (BF16),
  Gemma-4-26B-A4B (BF16 / AWQ-4bit), Gemma-4-31B (Quark-W8A8). vLLM uses
  `--attention-backend TRITON_ATTN` (only stable vLLM attention on gfx1151).
- **llama.cpp HIP** (`container: halo-llamacpp-node`, 28): Qwen3 4B/8B/14B/32B,
  Qwen3-30B-A3B, Qwen3.6-27B/35B-A3B, Gemma-4-26B-A4B, Llama-3.1-8B, MiMo-V2.5,
  Step-3.5-Flash — in BF16 / Q8_0 / Q4_K_M / UD-Q4_K_M / etc.

List them all with `./run-recipe.py --list`.

## Quick Start

```bash
./run-recipe.py --list
./run-recipe.py qwen3.6-35b-a3b-bf16-vllm --print     # show the launch command
MODELS_DIR=/models ./run-recipe.py qwen3-32b-q8-0-llamacpp
```

## Recipe schema

```yaml
recipe_version: "1"
name: <ShortName>
description: <what this serves>
model: /models/<dir-or-file>   # local path mounted into the container, or HF id
container: halo-vllm-node      # or halo-llamacpp-node
mods: []                       # list of mods/<dir> to apply
defaults:
  port: 8000
  host: 0.0.0.0
  nseq: 32                     # --max-num-seqs (vLLM) / -np (llama.cpp)
  gpu_memory_utilization: 0.85
env: {}
command: |
  vllm serve <model> --host {host} --port {port} ...
```

