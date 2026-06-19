# radeon-docker — vLLM &amp; llama.cpp containers for AMD Radeon (ROCm)

Build and run recent **vLLM** and **llama.cpp** on **AMD Radeon** GPUs via ROCm:
container builds, one-click recipes, and a solo launcher with the right GPU
passthrough. One Dockerfile per framework; the target GPU is selected by a
device profile in [`devices/`](devices/):

| Device | Arch | Status |
|---|---|---|
| `strix` — Strix Halo (Radeon 8060S iGPU, RDNA 3.5) | `gfx1151` | ✅ verified |
| `w7900` — Radeon PRO W7900 (RDNA3) | `gfx1100` | ⚠️ placeholder, unverified |
| `r9700` — Radeon AI PRO R9700 (RDNA4) | `gfx1200` | ⚠️ placeholder, base TBD |

The Dockerfiles, recipes, launch/build scripts, and the gfx1151 notes here are
**how the InferStation gfx1151 benchmark fleet (halo5 / halo6) actually builds
and serves models** — the build steps and serve commands are the same ones that
produce its daily results (180 vLLM serve runs on 2026-06-18 alone). Use
`build.sh` + `launch-cluster.sh --solo`, or `run-recipe.py <name>` to run a
recipe's serve command directly.

> **Read [`docs/GFX1151_NOTES.md`](docs/GFX1151_NOTES.md) first** — it has the
> hard-won facts (FLASH_ATTN is a dead end, no marlin MoE on ROCm, the C++23
> build fix, the FastAPI health-500 trap, etc.).

## Table of Contents

- [Quick Start](#quick-start)
- [1. Building the image](#1-building-the-image)
- [2. Running (solo)](#2-running-solo)
- [3. Recipes](#3-recipes)
- [4. Configuration](#4-configuration)
- [5. Mods and patches](#5-mods-and-patches)
- [6. Scripts](#6-scripts)
- [7. gfx1151 notes](#7-gfx1151-notes)
- [CHANGELOG](#changelog)

## Quick Start

```bash
git clone git@github.com:radeon-arena/radeon-docker.git
cd radeon-docker

# Build the vLLM image for Strix Halo (gfx1151): clones AMD's ROCm/vllm gfx11
# branch, applies the C++23 build fix, compiles HIP extensions. ~30-60 min cold.
./build.sh --framework vllm --device strix

# Serve a model (TRITON_ATTN is the only stable vLLM attention on gfx1151).
MODELS_DIR=/models ./launch-cluster.sh --solo -p 8000:8000 exec \
  vllm serve /models/Qwen3.6-35B-A3B \
    --host 0.0.0.0 --port 8000 \
    --max-num-seqs 32 --dtype bfloat16 \
    --attention-backend TRITON_ATTN
```

## 1. Building the image

```bash
./build.sh -f vllm      -d strix   # vLLM gfx11 branch  -> ghcr.io/radeon-arena/vllm:gfx1151
./build.sh -f vllm-main -d strix   # vLLM upstream main -> ghcr.io/radeon-arena/vllm-main:gfx1151
./build.sh -f llamacpp  -d strix   # llama.cpp (HIP)    -> ghcr.io/radeon-arena/llamacpp:gfx1151
./build.sh -f llamacpp  -d w7900   # same recipe, gfx1100 (RDNA3)
```

Default image tag is `ghcr.io/radeon-arena/<framework>:<gfx-arch>`. The old
`./build-and-copy.sh` still works as a strix-only shim.

- vLLM (default): [`dockerfiles/vllm/Dockerfile`](dockerfiles/vllm/Dockerfile) — base
  `rocm/vllm:rocm7.13.0_gfx1151_..._vllm_0.19.1`, builds a vLLM wheel from AMD's
  `ROCm/vllm` `gfx11` branch (gfx1151-tuned; also avoids the upstream-main AWQ
  MoE `tp_size` crash). The build applies the C++23 `std::in_range` fix.
- vLLM (upstream main): [`dockerfiles/vllm-main/Dockerfile`](dockerfiles/vllm-main/Dockerfile) — same base, but
  builds `vllm-project/vllm` **main**. Use this for models that need upstream
  main, notably **DiffusionGemma**. Mirrors InferStation's `vllm-rocm-halo-main`
  image.
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

**39 pre-configured serve commands** live in [`recipes/`](recipes/), generated
from the InferStation gfx1151 units and cross-checked against `runs.json` (each
corresponds to a config that produced a real decode result on halo5/halo6).
See [recipes/README.md](recipes/README.md). Examples:

- `qwen3.6-35b-a3b-bf16-vllm`, `qwen3.6-35b-a3b-awq-4bit-vllm`,
  `qwen3.6-35b-a3b-quark-w8a8-int8-vllm` (vLLM gfx11, TRITON_ATTN)
- `gemma-4-26b-a4b-it-awq-4bit-vllm`, `gemma-4-31b-it-quark-w8a8-int8-vllm` (vLLM gfx11)
- `qwen3-32b-q8-0-llamacpp`, `qwen3.6-35b-a3b-ud-q4-k-m-llamacpp`,
  `mimo-v2.5-ud-q2-k-xl-llamacpp` (llama.cpp HIP)
- `diffusiongemma-26b-a4b-bf16` / `-awq-int4` (vLLM **upstream-main** image; needs `--main` build)

List them all: `./run-recipe.py --list`.

## 4. Configuration

See [`.env.example`](.env.example).

## 5. Mods and patches

See [`mods/`](mods/) — `fix-gfx11-in-range` (build fix) and `force-triton-attn`
(runtime). See [mods/README.md](mods/README.md).

## 6. Scripts

- [`build.sh`](build.sh) — build an image for a framework × device.
- [`build-and-copy.sh`](build-and-copy.sh) — deprecated strix-only shim for `build.sh`.
- [`launch-cluster.sh`](launch-cluster.sh) — run a model (solo) with ROCm passthrough.
- [`hf-download.sh`](hf-download.sh) — download a model into `/models`.
- [`run-recipe.py`](run-recipe.py) / [`run-recipe.sh`](run-recipe.sh) — run a
  recipe's serve command via the solo launcher (`--print` to just show it).

## 7. gfx1151 notes

[`docs/GFX1151_NOTES.md`](docs/GFX1151_NOTES.md) — attention backends, MoE/marlin
on ROCm, AITER, skinny GEMM, the build fix, health-500, llama.cpp `-c`/`-np`
sizing, and more.

## DISCLAIMER

This repository is not affiliated with AMD or their subsidiaries. It is a
community project for running vLLM / llama.cpp on AMD Radeon GPUs via ROCm.

## CHANGELOG

### Unreleased
- vLLM + llama.cpp images for gfx1151; **39 serve recipes** (incl. DiffusionGemma BF16 + AWQ-INT4 on the upstream-main image) generated from the
  InferStation gfx1151 units and cross-checked against real `runs.json` results;
  solo launcher with ROCm passthrough; build fix for the gfx11 C++23
  `std::in_range`; gfx1151 notes.
