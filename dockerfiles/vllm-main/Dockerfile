# syntax=docker/dockerfile:1.6
#
# radeon-docker — vLLM (upstream main) for AMD Radeon (default: gfx1151 / ROCm).
#
# This is the SECOND vLLM line, alongside dockerfiles/vllm/Dockerfile (AMD's ROCm/vllm gfx11
# branch). It builds upstream `vllm-project/vllm` main instead.
#
# Why a separate main line:
#   - upstream main has model support the gfx11 branch lacks — notably
#     **DiffusionGemma** (vLLM PR #45163, merged into main).
#   - Tradeoff: it does NOT have the gfx11 branch's gfx1151-specific kernels,
#     and on ROCm its refactored MoE can hit `RoutedExperts has no attribute
#     tp_size` for some AWQ MoE models. So:
#       * gfx11 line (dockerfiles/vllm/Dockerfile, vllm image)  -> default, best for
#         AWQ/Quark/BF16 MoE serving on gfx1151.
#       * main line (this file, vllm-main image)     -> use for models that
#         require upstream main (DiffusionGemma).
#
# This mirrors the InferStation `vllm-rocm-halo-main` image (the
# diffusion-20260613 build that produced real DiffusionGemma results on halo).

# Same AMD gfx1151 base as the gfx11 line; the base's vLLM 0.19.1 is overwritten.
ARG VLLM_BASE=rocm/vllm:rocm7.13.0_gfx1151_ubuntu24.04_py3.13_pytorch_2.10.0_vllm_0.19.1

ARG VLLM_REPO=https://github.com/vllm-project/vllm.git
ARG VLLM_TAG=main
ARG PYTORCH_ROCM_ARCH=gfx1151

# Cache-bust for the `git clone` layer (main moves). Pass:
#   --build-arg CACHEBUST=$(git ls-remote https://github.com/vllm-project/vllm.git main | cut -f1)
ARG CACHEBUST=main

# =========================================================
# STAGE 1: build the vLLM wheel (upstream main) for gfx1151
# =========================================================
FROM ${VLLM_BASE} AS builder
ARG VLLM_REPO
ARG VLLM_TAG
ARG PYTORCH_ROCM_ARCH
ARG CACHEBUST

ENV PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH}
ENV MAX_JOBS=16
ENV CMAKE_BUILD_PARALLEL_LEVEL=16

WORKDIR /build

# Clone upstream main. CACHEBUST in the command text invalidates this layer
# whenever main's HEAD changes (otherwise the clone is cached and frozen).
RUN echo "cache-bust: ${CACHEBUST}" && \
    git clone --depth=1 --branch "${VLLM_TAG}" "${VLLM_REPO}" vllm

# NOTE: upstream main does not use the gfx11 branch's C++23 `std::in_range`, so
# the mods/fix-gfx11-in-range patch is NOT needed here. If a future main commit
# introduces a C++23-only symbol that breaks the HIP (C++20) device compile,
# apply an equivalent rewrite like that mod does.

# Build the wheel (HIP extensions compiled for gfx1151). ~30-60 min cold.
RUN cd vllm && \
    python3 -m pip install --no-cache-dir build && \
    python3 setup.py bdist_wheel && \
    mkdir -p /wheels && cp dist/*.whl /wheels/

# =========================================================
# STAGE 2: runtime
# =========================================================
FROM ${VLLM_BASE} AS runtime
ARG PYTORCH_ROCM_ARCH
ARG VLLM_TAG

LABEL io.amd.rocm.arch="${PYTORCH_ROCM_ARCH}" \
      org.opencontainers.image.description="vLLM (upstream ${VLLM_TAG}) for AMD Strix Halo (${PYTORCH_ROCM_ARCH})"

ENV PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH}

# Reinstall vLLM from the wheel just built (overwrites the base's 0.19.1).
COPY --from=builder /wheels/*.whl /tmp/wheels/
RUN python3 -m pip install --no-cache-dir --force-reinstall --no-deps /tmp/wheels/*.whl && \
    rm -rf /tmp/wheels

WORKDIR /workspace
CMD ["/bin/bash"]
