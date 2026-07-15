# Reproduce a Radeon Arena result

This repository contains the full runnable side of Radeon Arena: container builds,
serve recipes, the benchmark harness, and committed reference results. The public
website displays results; this repo produces them.

## What you need

- A real AMD Radeon machine with ROCm and Docker. The verified target today is
  Strix Halo / Radeon 8060S (`gfx1151`).
- Access to the model weights referenced by the recipe's `source` field. The
  current recipes use public Hugging Face repositories.
- Enough host memory and disk for the chosen model. Large BF16 and diffusion
  recipes can stage tens of GiB of weights.

GitHub-hosted runners cannot reproduce these results because they do not provide
Radeon GPUs. Use a local Radeon host or a self-hosted GitHub Actions runner.

## The files that define a run

For each benchmarked model, composable specs and the result are paired by the
matrix name:

```text
matrices/catalog.yaml           # test cases referencing the axes below
models/catalog.yaml             # model artifact/source/quantization
launches/catalog.yaml           # runtime/image/serve command
devices/<device>.yaml           # hardware identity/topology
benchmarking/<profile>.yaml     # workload matrix
results/strix/<matrix>.json     # reference result produced on gfx1151
recipes/<matrix>.yaml           # legacy compatibility view
```

The common benchmark profile is:

```text
benchmarking/halo-arena-v1.yaml # 512 input / 128 output, conc 1 / 4 / 16 / 32
```

When no explicit OCI image is declared, logical runtimes use these RadeonRun
defaults:

| recipe `container` | image |
|---|---|
| `llamacpp` | `ghcr.io/radeon-arena/halo-llamacpp:<tag>` |
| `vllm` | `ghcr.io/radeon-arena/halo-vllm:<tag>` |
| `vllm-main` | `ghcr.io/radeon-arena/halo-vllm-main:<tag>` |

These defaults are not required. Pass `--image` or declare `launch.image.ref`
to run from any OCI registry. `--registry` changes only the logical-runtime
fallback. See [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md).

## Local reproduction

List the available recipes:

```bash
./run-recipe.py --list
```

Print the resolved serve command without running anything:

```bash
./run-recipe.py qwen3-4b-q4-k-m-llamacpp --device halo --print
```

Run a full reproduce job locally. This ensures the image exists, downloads the
model into `MODELS_DIR`, starts the server, runs the benchmark profile, writes a
result JSON, and tears down the staged model:

```bash
MODELS_DIR=/models ./run-recipe.py qwen3-4b-q4-k-m-llamacpp \
  --device halo \
  --benchmark benchmarking/halo-arena-v1.yaml \
  --out /tmp/radeonrun-results/ \
  --build \
  --cleanup
```

If you also want to publish a freshly built image to GHCR, add `--push` after
logging in to `ghcr.io` with package write permission.

## GitHub Actions reproduction

The workflow at `.github/workflows/reproduce.yml` runs the same command on a
self-hosted runner:

```text
runs-on: [self-hosted, gfx1151]
python3 run-recipe.py <recipe> --device halo \
  --benchmark benchmarking/halo-arena-v1.yaml \
  --out results/ --build --push --cleanup
```

A runner should have labels like:

```text
self-hosted, Linux, X64, radeon, gfx1151, strix-halo, rocm
```

Trigger it from the GitHub UI or the API with inputs:

```json
{
  "recipe": "qwen3-4b-q4-k-m-llamacpp",
  "device": "halo"
}
```

The workflow uploads an artifact named `result-<recipe>` containing the result
JSON. Use that artifact as the new measurement output, not temporary files left
on the runner.

## Comparing with the reference result

The reference result is `results/strix/<recipe>.json`. The key comparison anchor
is C=1 single-stream TPOT / decode throughput, because it is least affected by
scheduler and batching differences. C=32 aggregate throughput is reported, but it
can vary with runtime version, cache state, and batching behavior.

A result is considered reproduced when the C=1 anchor is close and the command,
model, and benchmark profile match. High-concurrency differences should be
reported rather than hidden.

## Known caveats

- AWQ recipes use public third-party quantized weights when the original recipe
  weights are not public. Result JSON files mark this in `meta.reproduction`.
- DiffusionGemma requires the `vllm-main` image. Its generation is block-style,
  so TPOT can be null or less meaningful; compare decode throughput and latency
  with that context.
- Large BF16 GGUF models are often split under `BF16/*.gguf` in Hugging Face
  repos. `run-recipe.py` stages all shards and uses the first shard path.
- `gfx1151` vLLM should use `TRITON_ATTN`; ROCm `FLASH_ATTN` is not a reliable
  general path for these recipes.
- Do not mix the website with the runner: Radeon Arena displays results;
  `radeonrun` builds images and produces benchmark JSON.
