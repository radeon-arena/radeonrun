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

| Recipe | Engine | Notes |
|---|---|---|
| `qwen3.6-35b-a3b-bf16` | vLLM | TRITON_ATTN (only stable vLLM attention on gfx1151) |
| `gemma4-26b-a4b` | vLLM | AWQ-4bit, `--dtype float16` |
| `gemma4-31b-quark-w8a8` | vLLM | Quark W8A8 INT8, `--quantization quark` |
| `qwen3-30b-a3b-q4-llamacpp` | llama.cpp | HIP backend, `-fa on` |

## Quick Start (intended)

```bash
./run-recipe.sh --list
./run-recipe.sh qwen3.6-35b-a3b-bf16 --solo
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

