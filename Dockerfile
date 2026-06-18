# syntax=docker/dockerfile:1.6
#
# SCAFFOLD Dockerfile for halo-vllm-docker (AMD Strix Halo / gfx1151 / ROCm).
#
# This is a multi-stage skeleton: the ROCm base image, kernel/attention
# backends, and vLLM build are TODO and must be chosen for gfx1151 and verified
# on real hardware before this can be considered working. Do NOT assume
# `docker build` succeeds yet.

# TODO(ROCm): pick and pin a real ROCm base. Candidates to evaluate on gfx1151:
#   - rocm/vllm-dev:<tag>
#   - rocm/dev-ubuntu-24.04:7.2-complete  (then build vLLM from source)
ARG ROCM_IMAGE=rocm/dev-ubuntu-24.04:7.2-complete

# Limit build parallelism to reduce OOM situations
ARG BUILD_JOBS=16

# =========================================================
# STAGE 1: Base Build Image
# =========================================================
FROM ${ROCM_IMAGE} AS base

ARG BUILD_JOBS
ENV MAX_JOBS=${BUILD_JOBS}
ENV CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS}
ENV NINJAFLAGS="-j${BUILD_JOBS}"
ENV MAKEFLAGS="-j${BUILD_JOBS}"

# Non-interactive apt, allow global pip on Ubuntu 24.04
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV PIP_CACHE_DIR=/root/.cache/pip

# TODO(ROCm): target architecture for Strix Halo.
ENV PYTORCH_ROCM_ARCH=gfx1151
# ENV HSA_OVERRIDE_GFX_VERSION=11.5.1   # set/verify on hardware

# TODO: system deps (git, build tooling, etc.)
# RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates && rm -rf /var/lib/apt/lists/*

# =========================================================
# STAGE 2: vLLM build / install
# =========================================================
FROM base AS vllm-build

# TODO(ROCm): either
#   (a) pip install a prebuilt ROCm vLLM wheel for gfx1151, or
#   (b) git clone vLLM and build the HIP extensions from source.
# The right approach for ROCm/gfx1151 must be decided and tested. Leaving
# unimplemented on purpose.
#
# RUN echo "TODO: install/build vLLM for ROCm gfx1151"

# =========================================================
# STAGE 3: Runtime
# =========================================================
FROM base AS runtime

WORKDIR /workspace

# TODO: copy built artifacts from the vllm-build stage, set entrypoint, etc.
# COPY --from=vllm-build /opt/venv /opt/venv

# The launcher clears the entrypoint by default; keep an interactive shell so
# initialization scripts can run before vLLM is started.
CMD ["/bin/bash"]
