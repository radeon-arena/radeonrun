# radeonrun Recipe Reference

A recipe is a YAML file that fully describes how to serve **and reproduce** an
inference workload on AMD Radeon (ROCm / gfx11xx): model, container, runtime,
configuration, and the exact command. Every number on the **Radeon Arena**
leaderboard is backed by a recipe + a [benchmark profile](benchmarking/README.md),
so it is reproducible.

```bash
python run-recipe.py recipes/qwen3.6-35b-a3b-bf16-vllm.yaml            # serve
python run-recipe.py recipes/qwen3.6-35b-a3b-bf16-vllm.yaml \
    --benchmark benchmarking/halo-arena-v1.yaml --out results/        # benchmark
```

### Self-contained runs (setup → run → teardown)

`run-recipe.py` stages **both halves** a run needs — the serve **image** and the
**model** — so a recipe reproduces from nothing but this repo (+ a HuggingFace
pull), with no pre-staged images or models on the host:

- **Image**: local image → `docker pull` → **build from `dockerfiles/`** (via
  `build.sh`, which names the image exactly as the run expects, so a local build
  needs no registry at all).
- **Model**: downloaded from the recipe's `source` into `$MODELS_DIR`.

```bash
python run-recipe.py qwen3-8b-q4-k-m-llamacpp --setup-only           # prepare image + model, then stop
python run-recipe.py qwen3-8b-q4-k-m-llamacpp \
    --benchmark benchmarking/halo-arena-v1.yaml --out results/ \
    --cleanup                                                        # setup → serve → bench → delete model
```

Setup runs automatically before a serve/benchmark unless you pass `--no-setup`.
Image source control:

| Flag         | Image behaviour                                                                  |
|--------------|----------------------------------------------------------------------------------|
| *(default)*  | local → pull → build from `dockerfiles/`                                          |
| `--build`    | build from source, skip the registry pull                                        |
| `--no-build` | only use a local or pulled image (never build)                                   |
| `--push`     | after a build, push `:commit` + `:latest` to ghcr (needs `docker login ghcr.io`) |

With `--push`, a runner that builds an image syncs it back to
`ghcr.io/radeon-arena/<device>-<engine>` so the next run (here or on any other
runner) just pulls it instead of rebuilding. Push failures only warn — the
freshly built image is still used locally.

`--cleanup` deletes the staged model afterwards to free disk; gated/private
model repos use `--hf-token` (or `$HF_TOKEN`). Models land under `$MODELS_DIR`
(default `/models`), which `launch-cluster.sh` bind-mounts to `/models` in the
container.

## Minimal recipe

```yaml
recipe_version: "2"
model: Qwen/Qwen3-8B
runtime: vllm
container: vllm
defaults:
  port: 8000
  tensor_parallel: 1
```

When `command` is omitted the runtime generates it from `defaults`.

---

## Field reference

### Core

| Field             | Type   | Required    | Default         | Description |
|-------------------|--------|-------------|-----------------|-------------|
| `recipe_version`  | string | no          | `"2"`           | Schema version. `"2"` is current. |
| `name`            | string | no          | derived         | Short identifier. |
| `model`           | string | **yes**     | —               | In-container path the serve command reads: a directory (`/models/Qwen3.6-35B-A3B`) or a `.gguf` file (`/models/Qwen3-8B/Qwen3-8B-Q4_K_M.gguf`). The runner stages it here from `source`. (May also be a bare HF id when no `source` is given.) |
| `source`          | string | for staging | `null`          | HF repo the runner downloads the model from (`run-recipe.py --setup`). The fetch shape is inferred from `model`: a `.gguf` path pulls that file (or every shard of a split gguf); a directory pulls the whole repo. Omit only when the model is already present on the host. |
| `model_revision`  | string | no          | `null`          | Pin `source` to an HF revision (branch, tag, or **commit hash**) for byte-identical, reproducible deployments. |
| `runtime`         | string | no          | auto-detected   | `vllm` or `llama-cpp`. See [Runtime resolution](#runtime-resolution). |
| `container`       | string | recommended | runtime default | Logical engine (`vllm`, `vllm-main`, `llamacpp`), resolved per `--device`/`--tag` to `ghcr.io/radeon-arena/<device>-<engine>:<commit>`; or a pinned `repo@sha256:…`. Set `image_tag` to pin a build commit. |
| `mods`            | list   | no          | `[]`            | Patch directories applied before launch (e.g. `mods/fix-gfx11-in-range`). |

GGUF models use colon syntax (`repo:quant`) to download only the matching
quantization files.

### Topology

| Field          | Type | Default | Description |
|----------------|------|---------|-------------|
| `min_nodes`    | int  | `1`     | Minimum hosts. `> 1` forces cluster mode. |
| `max_nodes`    | int  | `null`  | Maximum hosts. `1` forces solo. |

On Strix Halo (1 GPU per node) `tensor_parallel: N` = N hosts.

### Configuration

| Field      | Type   | Default | Description |
|------------|--------|---------|-------------|
| `defaults` | map    | `{}`    | Default values for serve flags. CLI overrides win. |
| `env`      | map    | `{}`    | Container environment variables (`VLLM_*`, `HIP_*`, etc.). |
| `command`  | string | `null`  | Command template. `{key}` placeholders resolved from `defaults`. |

### Metadata (informational; not passed to the runtime)

```yaml
metadata:
  description: "Qwen3.6-35B-A3B — BF16 on Strix Halo"
  maintainer: "you <you@example.com>"
  model_params: 35B
  model_dtype: bfloat16     # float16, bfloat16, fp8, int8, awq4, q4_k_m, q8_0, ...
  quantization: none        # awq, gptq, fp8, quark-w8a8, compressed-tensors, none
  measured:
    gpu: "Radeon 8060S (Strix Halo, gfx1151)"
    profile: halo-arena-v1
    decode_toks_per_s: 124.3
```

---

## Runtime resolution

| Condition                                | Resolved runtime |
|------------------------------------------|------------------|
| `runtime: vllm` or empty                 | `vllm`           |
| Command starts with `llama-server`       | `llama-cpp`      |
| `runtime: llama-cpp`                     | `llama-cpp`      |

Explicit `runtime` always wins.

---

## Defaults keys: vLLM vs llama.cpp

The same logical key maps to a different CLI flag per runtime. This is the core
difference between a vLLM and a llama.cpp recipe.

| Key                      | vLLM (`vllm serve`)        | llama.cpp (`llama-server`)        | Description |
|--------------------------|----------------------------|-----------------------------------|-------------|
| `port` / `host`          | `--port` / `--host`        | `--port` / `--host`               | Bind address |
| `tensor_parallel`        | `-tp`                      | `--split-mode row`                | TP degree (= node count) |
| `pipeline_parallel`      | `-pp`                      | `--split-mode layer`              | PP degree |
| `max_model_len`          | `--max-model-len`          | `--ctx-size` / `-c`               | Max sequence / context length |
| `gpu_memory_utilization` | `--gpu-memory-utilization` | —                                 | GPU memory fraction |
| `max_num_seqs`           | `--max-num-seqs`           | `-np` (parallel slots)            | Max concurrent sequences |
| `max_num_batched_tokens` | `--max-num-batched-tokens` | —                                 | Batch token budget |
| `dtype`                  | `--dtype`                  | (baked into GGUF)                 | Model dtype |
| `quantization`           | `--quantization`           | (baked into GGUF)                 | Quantization method |
| `kv_cache_dtype`         | `--kv-cache-dtype`         | `--cache-type-k` / `-ctk`         | KV cache dtype |
| `attention_backend`      | `--attention-backend`      | `--flash-attn on/off`             | Attention kernel |
| `n_gpu_layers`           | —                          | `--n-gpu-layers` / `-ngl`         | Layers offloaded to GPU |
| `speculative_config`     | `--speculative-config`     | `--model-draft` / `-md`           | Speculative decoding |
| `served_model_name`      | `--served-model-name`      | `--alias`                         | Model name in the API |

Any key may appear in `defaults`; unknown keys are passed straight through to
`{key}` substitution in the command template.

> **vLLM on gfx11 (Strix Halo) note:** use `--attention-backend TRITON_ATTN`.
> `FLASH_ATTN` is a dead end on gfx1151 (see [docs/GFX1151_NOTES.md](docs/GFX1151_NOTES.md)).

---

## vLLM recipe example (full config)

```yaml
recipe_version: "2"
model: /models/Qwen3.6-35B-A3B
runtime: vllm
container: vllm

metadata:
  description: Qwen3.6-35B-A3B (BF16) on Strix Halo

defaults:
  port: 8000
  host: 0.0.0.0
  tensor_parallel: 1
  max_num_seqs: 32
  dtype: bfloat16
  attention_backend: TRITON_ATTN     # required on gfx1151

env:
  VLLM_USE_TRITON_FLASH_ATTN: "1"

command: |
  vllm serve {model} \
    --host {host} --port {port} \
    --max-num-seqs {max_num_seqs} \
    --dtype {dtype} \
    --attention-backend {attention_backend} \
    -tp {tensor_parallel}
```

### Speculative decoding (MTP)

Like Spark Arena's `*-mtp-*` recipes, multi-token prediction is opt-in via a
single field. Adding it is the *only* difference between a base and an MTP recipe:

```yaml
defaults:
  speculative_config: '{"method": "mtp", "num_speculative_tokens": 2}'
command: |
  vllm serve {model} ... \
    --speculative-config '{speculative_config}'
```

---

## llama.cpp recipe example (GGUF)

```yaml
recipe_version: "2"
model: /models/Qwen3.6-35B-A3B/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
runtime: llama-cpp
max_nodes: 1
container: llamacpp

metadata:
  description: Qwen3.6-35B-A3B (UD-Q4_K_M) on Strix Halo

defaults:
  port: 8000
  host: 0.0.0.0
  n_gpu_layers: 999
  ctx: 32768
  nseq: 32

command: |
  llama-server -m {model} \
    -ngl {n_gpu_layers} -fa on \
    --host {host} --port {port} \
    -c {ctx} -np {nseq}
```

Key differences from vLLM: GGUF model file (quantization baked in), `n_gpu_layers`
instead of TP, `--flash-attn` instead of an attention-backend flag, and far fewer
engine-tuning knobs.

---

## Reproducing a leaderboard number

```bash
# 1. Pick the exact recipe behind a result
#    (recipe files live in this repo; the static site reads results/bundle.json)

# 2. Serve + benchmark against the same standardized profile
python run-recipe.py <recipe-name> \
  --benchmark benchmarking/halo-arena-v1.yaml \
  --out results/

# 3. Compare results/<recipe>.json against the published number
```

Because the recipe pins the model, container, command and engine flags, and the
profile pins the test grid (shapes, depths, concurrency, repeats), the numbers
should match within run-to-run noise.

**Worked examples:** all 40 recipes have already been reproduced this way — the
measured result JSONs are in [`results/strix/`](results/strix/) and the verdicts
in [`docs/REPRODUCTION.md`](docs/REPRODUCTION.md).

The self-hosted GitHub Actions workflow `.github/workflows/reproduce.yml` runs
the same command, commits `results/<device>/<recipe>.json`, regenerates
`results/index.json` and `results/bundle.json`, and pushes those generated files
back to the triggering branch. `validate-results.yml` checks recipe/result
structure and bundle coverage on PRs and pushes.
