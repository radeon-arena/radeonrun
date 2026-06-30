# radeonrun — build, serve &amp; benchmark vLLM / llama.cpp on AMD Radeon (ROCm)

A toolkit for running recent **vLLM** and **llama.cpp** on **AMD Radeon** GPUs
via ROCm: container builds, one-click serve recipes, a solo launcher with the
right GPU passthrough, and a reproducible benchmark harness. One Dockerfile per
framework; the target GPU is selected by a device profile in [`devices/`](devices/):

| Device | Arch | Status |
|---|---|---|
| `halo` — Strix Halo (Radeon 8060S iGPU, RDNA 3.5) | `gfx1151` | ✅ verified |
| `w7900` — Radeon PRO W7900 (RDNA3) | `gfx1100` | ⚠️ placeholder, unverified |
| `r9700` — Radeon AI PRO R9700 (RDNA4) | `gfx1200` | ⚠️ placeholder, base TBD |

Images are named **`ghcr.io/radeon-arena/<device>-<framework>:<commit>`** — the
device is in the name and the tag is the upstream build commit (so a result can
pin a byte-reproducible image, e.g. `halo-llamacpp:fe7c8b2414`); `:latest` is
also published for convenience.

The Dockerfiles, recipes, launch/build scripts, and the gfx1151 notes here are
**the exact build steps and serve commands that produce the leaderboard numbers**
— every recipe is independently measured on real gfx1151 hardware, see
[8. Reproduction](#8-reproduction). For a step-by-step reproduction flow, start
with [`REPRODUCE.md`](REPRODUCE.md). Use `build.sh` + `launch-cluster.sh --solo`,
or `run-recipe.py <name>` to run a recipe's serve command directly.

> **Read [`docs/GFX1151_NOTES.md`](docs/GFX1151_NOTES.md) first** — it has the
> hard-won facts (FLASH_ATTN is a dead end, no marlin MoE on ROCm, the C++23
> build fix, the FastAPI health-500 trap, etc.).

## Table of Contents

- [Quick Start](#quick-start)
- [Reproduce a Result](#reproduce-a-result)
- [1. Building the image](#1-building-the-image)
- [2. Running (solo)](#2-running-solo)
- [3. Recipes](#3-recipes)
- [4. Configuration](#4-configuration)
- [5. Mods and patches](#5-mods-and-patches)
- [6. Scripts](#6-scripts)
- [7. gfx1151 notes](#7-gfx1151-notes)
- [8. Reproduction](#8-reproduction)
- [CHANGELOG](#changelog)

## Quick Start

```bash
git clone git@github.com:radeon-arena/radeonrun.git
cd radeonrun

# Build the vLLM image for Strix Halo (gfx1151): clones AMD's ROCm/vllm gfx11
# branch, applies the C++23 build fix, compiles HIP extensions. ~30-60 min cold.
./build.sh --framework vllm --device halo

# Serve a model (TRITON_ATTN is the only stable vLLM attention on gfx1151).
MODELS_DIR=/models ./launch-cluster.sh --solo -p 8000:8000 exec \
  vllm serve /models/Qwen3.6-35B-A3B \
    --host 0.0.0.0 --port 8000 \
    --max-num-seqs 32 --dtype bfloat16 \
    --attention-backend TRITON_ATTN
```

## Reproduce a Result

Use [`REPRODUCE.md`](REPRODUCE.md) for the complete flow. In short:

```bash
MODELS_DIR=/models ./run-recipe.py qwen3-4b-q4-k-m-llamacpp \
  --device halo \
  --benchmark benchmarking/halo-arena-v1.yaml \
  --out /tmp/radeonrun-results/ \
  --build \
  --cleanup
```

The matching reference result is committed under
`results/strix/qwen3-4b-q4-k-m-llamacpp.json`. The GitHub Actions workflow
`.github/workflows/reproduce.yml` runs the same command on a self-hosted
`gfx1151` Radeon runner, commits the reproduced result back to
`results/<device>/`, regenerates `results/index.json` and `results/bundle.json`,
and pushes those files back to the triggering branch. A `result-<recipe>`
artifact is still uploaded for audit/debugging.

## 1. Building the image

```bash
./build.sh -f vllm      -d halo    # vLLM gfx11 branch  -> ghcr.io/radeon-arena/halo-vllm:<commit>
./build.sh -f vllm-main -d halo    # vLLM upstream main -> ghcr.io/radeon-arena/halo-vllm-main:<commit>
./build.sh -f llamacpp  -d halo    # llama.cpp (HIP)    -> ghcr.io/radeon-arena/halo-llamacpp:<commit>
./build.sh -f llamacpp  -d w7900   # same recipe, gfx1100 (RDNA3) -> w7900-llamacpp:<commit>
```

Each build tags both `:<commit>` (the upstream build commit) and `:latest`. The
old `./build-and-copy.sh` still works as a halo-only shim.

- vLLM (default): [`dockerfiles/vllm/Dockerfile`](dockerfiles/vllm/Dockerfile) — base
  `rocm/vllm:rocm7.13.0_gfx1151_..._vllm_0.19.1`, builds a vLLM wheel from AMD's
  `ROCm/vllm` `gfx11` branch (gfx1151-tuned; also avoids the upstream-main AWQ
  MoE `tp_size` crash). The build applies the C++23 `std::in_range` fix.
- vLLM (upstream main): [`dockerfiles/vllm-main/Dockerfile`](dockerfiles/vllm-main/Dockerfile) — same base, but
  builds `vllm-project/vllm` **main**. Use this for models that need upstream
  main, notably **DiffusionGemma**.
- llama.cpp: [`dockerfiles/llamacpp/Dockerfile`](dockerfiles/llamacpp/Dockerfile) — base
  `rocm/dev-ubuntu-24.04:7.2.1-complete` (ROCm version must match the host KFD
  driver), `-DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1151`.

## 2. Running (solo)

`launch-cluster.sh --solo` runs the container with the verified ROCm GPU
passthrough (`--device /dev/kfd --device /dev/dri --group-add video
--security-opt seccomp=unconfined --ipc host`), mounts `/models` and the HF
cache, clears the entrypoint, and execs your command.

Multi-node / cluster mode is **not** implemented — Strix Halo is a single-GPU
APU and our experience is single-node only.

## 3. Recipes

**40 pre-configured serve commands** live in [`recipes/`](recipes/), each
independently measured on real gfx1151 hardware (the `results/strix/` JSON is the
source of truth). See [recipes/README.md](recipes/README.md). Examples:

- `qwen3.6-35b-a3b-bf16-vllm`, `qwen3.6-35b-a3b-awq-4bit-vllm`,
  `qwen3.6-35b-a3b-quark-w8a8-int8-vllm` (vLLM gfx11, TRITON_ATTN)
- `gemma-4-26b-a4b-it-awq-4bit-vllm`, `gemma-4-31b-it-quark-w8a8-int8-vllm` (vLLM gfx11)
- `qwen3-32b-q8-0-llamacpp`, `qwen3.6-35b-a3b-ud-q4-k-m-llamacpp`,
  `mimo-v2.5-ud-q2-k-xl-llamacpp` (llama.cpp HIP)
- `diffusiongemma-26b-a4b-bf16` / `-awq-int4` (vLLM **upstream-main** image; needs `--main` build)

List them all: `./run-recipe.py --list`.

All 40 recipes have been independently re-run on real gfx1151 hardware — see
[8. Reproduction](#8-reproduction).

Radeon Arena (the website) reads the browser-facing bundle committed here:
[`results/bundle.json`](results/bundle.json). [`results/index.json`](results/index.json)
lists the per-device result files included in that bundle.

## 4. Configuration

See [`.env.example`](.env.example).

## 5. Mods and patches

See [`mods/`](mods/) — `fix-gfx11-in-range` (build fix) and `force-triton-attn`
(runtime). See [mods/README.md](mods/README.md).

## 6. Scripts

- [`build.sh`](build.sh) — build an image for a framework × device.
- [`build-and-copy.sh`](build-and-copy.sh) — deprecated halo-only shim for `build.sh`.
- [`launch-cluster.sh`](launch-cluster.sh) — run a model (solo) with ROCm passthrough.
- [`hf-download.sh`](hf-download.sh) — download a model into `/models`.
- [`run-recipe.py`](run-recipe.py) / [`run-recipe.sh`](run-recipe.sh) — run a
  recipe's serve command via the solo launcher (`--print` to just show it).

## 7. gfx1151 notes

[`docs/GFX1151_NOTES.md`](docs/GFX1151_NOTES.md) — attention backends, MoE/marlin
on ROCm, AITER, skinny GEMM, the build fix, health-500, llama.cpp `-c`/`-np`
sizing, and more.

## 8. Reproduction

Every recipe was independently re-run on real Strix Halo (gfx1151) hardware with
the repo's own `benchmarking/halo-arena-v1.yaml` profile. The measured numbers
are committed as native result files under [`results/strix/`](results/strix/)
(one JSON per recipe, each with a `meta.reproduction` block recording the actual
weights used and a one-line verdict).

See [`docs/REPRODUCTION.md`](docs/REPRODUCTION.md) for the method, the per-recipe
verdict table, and the cross-cutting findings (e.g. AWQ runs use public
third-party weights, INT8 shows no decode speedup on dense gfx1151, DiffusionGemma
needs the `vllm-main` image).

To run a reproduction yourself, use [`REPRODUCE.md`](REPRODUCE.md). It covers the
local CLI path, the self-hosted GitHub Actions path, where images/recipes/results
live, and how to compare your output with `results/strix/*.json`.

The GitHub Actions path is closed-loop: `reproduce.yml` runs the selected recipe,
writes `results/<device>/<recipe>.json`, regenerates `results/index.json` and
`results/bundle.json`, and pushes those files back to the branch. The
`validate-results.yml` workflow checks recipe structure, result structure, and
bundle coverage on pushes and pull requests.

> **Two separate projects** — *radeonrun* (this repo: images + recipes + the
> benchmark harness that **produces** the numbers) and *Radeon Arena* (the website
> that **displays** them) are distinct. The numbers come from this repo's own runs
> under [`results/strix/`](results/strix/) and are served to the static website
> through [`results/bundle.json`](results/bundle.json), not from a website
> database.

## DISCLAIMER

This repository is not affiliated with AMD or their subsidiaries. It is a
community project for running vLLM / llama.cpp on AMD Radeon GPUs via ROCm.

## CHANGELOG

### Unreleased
- vLLM + llama.cpp images for gfx1151; **40 serve recipes** (incl. DiffusionGemma
  BF16 + AWQ-INT4 on the upstream-main image), each independently measured on real
  gfx1151 hardware; solo launcher with ROCm passthrough; build fix for the gfx11
  C++23 `std::in_range`; gfx1151 notes.
- All 40 recipes independently reproduced on real gfx1151 hardware →
  [`results/strix/`](results/strix/) + [`docs/REPRODUCTION.md`](docs/REPRODUCTION.md).
- `reproduce.yml` now auto-commits reproduced result JSON and regenerated
  `results/index.json` / `results/bundle.json`; `validate-results.yml` verifies
  recipe/result structure and bundle coverage.
