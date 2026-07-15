# Legacy recipe template

> New configurations should add or reuse entries under `models/`, `launches/`,
> `devices/`, and `matrices/`. See
> [`../docs/CONFIGURATION.md`](../docs/CONFIGURATION.md). This template remains
> for integrations that still provide a single flat recipe.

Copy the YAML block below into `recipes/<recipe-id>.yaml`, then replace every
`<...>` placeholder. Keep this file as Markdown: files matching `recipes/*.yaml`
are treated as real runnable recipes by `run-recipe.py` and the recipe browser.

## What belongs in a recipe

A legacy recipe combines model and launch configuration in one file.
Benchmark workload configuration lives outside the recipe, in
`benchmarking/halo-arena-v1.yaml` / `benchmarking/halo-arena-v2.yaml`, with
per-model limits in `benchmarking/model-contexts.yaml`.

Serve-side fields in a recipe:

- `source`, `model`: where the runner stages the model from and where the
  container reads it.
- `container`, `image_tag`: runtime image line and pinned build.
- `defaults`: serve defaults such as `port`, `host`, `nseq`, and `ctx`.
- `env`, `model_patches`: runtime environment and staged-model metadata patches.
- `command`: the actual `vllm serve` or `llama-server` command template.

Benchmark-side fields outside the recipe:

- `depth`, `pp`, `tg`, `concurrency`, `warmup`, `runs`, `prefix_caching` in the
  benchmark profile.
- `model_ctx` / `benchmark_ctx` in `benchmarking/model-contexts.yaml`, or a
  recipe-local `benchmark_ctx` only when this recipe intentionally serves below
  the model's theoretical context.

## YAML skeleton

```yaml
# Recipe: <Model name> (<quantization>) on Strix Halo - <runtime>
# Replace placeholders before committing. Do not claim measured performance until
# a real halo-arena profile result has been written under results/strix/.

recipe_version: "2"
metadata:
  description: <runtime> serving <model> (<quantization>) on Strix Halo
  maintainer: radeon-arena
  quantization: <BF16|Q8_0|Q4_K_M|AWQ-4bit|Quark-W8A8-INT8|...>
  # Add this block only after benchmarking on real gfx1151 hardware:
  # measured:
  #   gpu: "Radeon 8060S (Strix Halo, gfx1151)"
  #   profile: halo-arena-v1
  #   decode_toks_per_s: <best-decode-toks-per-s>

name: <recipe-display-name>
description: <runtime> serving <model> (<quantization>) on Strix Halo
model: /models/<local-model-path-or-gguf-file>
source: <hf-org>/<hf-repo>
# model_revision: <optional-hf-revision>

# Runtime image line. `run-recipe.py --device halo` resolves these to:
#   vllm      -> ghcr.io/radeon-arena/halo-vllm:<image_tag>
#   vllm-main -> ghcr.io/radeon-arena/halo-vllm-main:<image_tag>
#   llamacpp  -> ghcr.io/radeon-arena/halo-llamacpp:<image_tag>
container: <vllm|vllm-main|llamacpp>
image_tag: <f5fa386fe|92221485a|fe7c8b2414|new-pinned-tag>

# Optional per-request benchmark cap for recipes intentionally served below the
# source model's full context. Prefer adding model_ctx/benchmark_ctx to
# benchmarking/model-contexts.yaml when the value is model-wide.
# benchmark_ctx: <tokens>

mods: []

defaults:
  port: 8000
  host: 0.0.0.0
  nseq: 32
  ctx: <serve-context>

env: {}

# Optional model config patch block. Keep absent unless the model needs it.
# model_patches:
#   - type: set_quant_config
#     values:
#       quant_method: gptq
#       desc_act: false

command: |
  <serve command using {host}, {port}, {nseq}, {ctx} as needed>
```

## vLLM command example

```yaml
command: |
  vllm serve /models/<hf-model-dir> --host {host} --port {port} --served-model-name <alias> --max-num-seqs {nseq} --dtype bfloat16 --attention-backend TRITON_ATTN --max-model-len {ctx} --gpu-memory-utilization 0.85
```

For vLLM, `{ctx}` is a per-request `--max-model-len` value.

## llama.cpp command example

```yaml
command: |
  llama-server -m /models/<model-dir>/<model-file>.gguf -ngl 999 -fa on --host {host} --port {port} -c {ctx} -np {nseq}
```

For llama.cpp, `run-recipe.py` may render `{ctx}` as total KV context when a
benchmark profile is supplied, because llama.cpp `-c` is shared across `-np`
slots rather than being per request.

## Checks before committing

```bash
python3 - <<'PY'
from pathlib import Path
import yaml
for path in Path('recipes').glob('*.yaml'):
    yaml.safe_load(path.read_text())
print('recipe yaml parse ok')
PY

python3 run-recipe.py <recipe-id> --device halo --no-setup --no-build --print
python3 scripts/build-recipe-site.py --out /tmp/radeonrun-recipes-check.html
```