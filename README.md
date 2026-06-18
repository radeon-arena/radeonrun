# vLLM Docker Optimized for AMD Strix Halo (scaffold)

> **Status: SCAFFOLD.** This repository currently contains only the directory
> structure and file skeletons. The ROCm/Halo implementation has **not** been
> written or hardware-validated yet. Files marked `SCAFFOLD` / `TODO` are
> placeholders, not working code. See [NOTICE](NOTICE).

A community effort to build and run recent vLLM on **AMD Strix Halo**
(Radeon 8060S / `gfx1151` / ROCm 7.2.x), mirroring the workflow of the
NVIDIA DGX Spark project [`spark-vllm-docker`](https://github.com/eugr/spark-vllm-docker)
(MIT, © Eugene Rakhmatulin) that this scaffold is derived from.

## Table of Contents

- [DISCLAIMER](#disclaimer)
- [QUICK START](#quick-start)
- [1. Building the Docker Image](#1-building-the-docker-image)
- [2. Launching (solo / cluster)](#2-launching-solo--cluster)
- [3. Running the Container (Manual)](#3-running-the-container-manual)
- [4. Configuration Details](#4-configuration-details)
- [5. Mods and Patches](#5-mods-and-patches)
- [6. Launch Scripts](#6-launch-scripts)
- [7. Cluster mode for inference](#7-cluster-mode-for-inference)
- [8. Model Loading](#8-model-loading)
- [9. Benchmarking](#9-benchmarking)
- [10. Downloading Models](#10-downloading-models)
- [CHANGELOG](#changelog)

## DISCLAIMER

This repository is not affiliated with AMD or their subsidiaries. It is a
community scaffold aimed at helping Strix Halo users set up and run recent
versions of vLLM on ROCm.

It is **derived from** [`spark-vllm-docker`](https://github.com/eugr/spark-vllm-docker)
(MIT License, © 2026 Eugene Rakhmatulin), which targets NVIDIA DGX Spark.
The original project and its author are not affiliated with this port.

## QUICK START

> **TODO:** every command below is a placeholder showing the *intended* shape.
> None of it works until the scaffold is filled in and tested on real hardware.

### Build

```bash
git clone git@github.com:JoursBleu/halo-vllm-docker.git
cd halo-vllm-docker

# TODO: build the ROCm container (see Dockerfile, currently a skeleton)
./build-and-copy.sh
```

### Run (solo)

```bash
# TODO: launch a model on a single Strix Halo node
./launch-cluster.sh --solo exec \
  vllm serve <model> --port 8000 --host 0.0.0.0
```

## 1. Building the Docker Image

TODO. The [`Dockerfile`](Dockerfile) skeleton sketches the ROCm base + vLLM
build stages but is not yet implemented. Unlike the CUDA upstream, the base
image, attention backends, and quantization kernels must be chosen for
`gfx1151` and verified on hardware.

## 2. Launching (solo / cluster)

TODO. See [`launch-cluster.sh`](launch-cluster.sh). Strix Halo is typically a
single-node APU; the cluster path is kept for structural parity and is TODO.

## 3. Running the Container (Manual)

TODO.

## 4. Configuration Details

See [`.env.example`](.env.example) for the configuration variables.

## 5. Mods and Patches

Per-model fixes live under [`mods/`](mods/). Each mod is a directory with a
`run.sh` applied at container/launch time. See [mods/README.md](mods/README.md).

## 6. Launch Scripts

- [`build-and-copy.sh`](build-and-copy.sh) — build the image (and optionally
  copy it to other nodes).
- [`launch-cluster.sh`](launch-cluster.sh) — run the container (solo or cluster).
- [`autodiscover.sh`](autodiscover.sh) — discover nodes / interfaces.
- [`hf-download.sh`](hf-download.sh) — download models from HuggingFace.
- [`run-recipe.py`](run-recipe.py) / [`run-recipe.sh`](run-recipe.sh) — one-click
  recipe runner.

## 7. Cluster mode for inference

TODO.

## 8. Model Loading

TODO.

## 9. Benchmarking

TODO.

## 10. Downloading Models

See [`hf-download.sh`](hf-download.sh) (skeleton).

## CHANGELOG

### Unreleased
- Initial scaffold mirroring `spark-vllm-docker`'s structure, retargeted at
  AMD Strix Halo (`gfx1151` / ROCm). No working implementation yet.
