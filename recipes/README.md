# Recipes

Recipes are a **one-click** way to deploy a model with pre-configured settings.Each recipe is a YAML file specifying:

- HuggingFace model to download
- Container image and build arguments
- Required mods/patches
- Default parameters (port, host, tensor parallelism, etc.)
- Environment variables
- The vLLM serve command

> **SCAFFOLD:** the example recipes here are skeletons targeting ROCm/Strix
> Halo and have **not** been validated on hardware. Flags such as `--attention-backend`,
> `--quantization`, and `--kv-cache-dtype` must be chosen for `gfx1151` and
> tested before a recipe can be trusted.

## Quick Start (intended)

```bash
./run-recipe.sh --list
./run-recipe.sh qwen3.6-35b-a3b-fp8 --solo
./run-recipe.sh qwen3.6-35b-a3b-fp8 --solo --setup
```

## Recipe schema

```yaml
recipe_version: "1"
name: <ShortName>
description: <what this serves>
model: <org/model>          # HF id, used by --download-model
container: halo-vllm-node    # image to use
mods: []                     # list of mods/<dir> to apply
defaults:
  port: 8000
  host: 0.0.0.0
  tensor_parallel: 1
  gpu_memory_utilization: 0.8
  max_model_len: 32768
  max_num_batched_tokens: 8192
env: {}
command: |
  vllm serve <model> --host {host} --port {port} ...
```
