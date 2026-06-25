# syntax=docker/dockerfile:1.6
#
# radeonrun — vLLM for AMD Radeon (default: Strix Halo / Radeon 8060S / gfx1151 / ROCm).
#
# This recreates, in a single multi-stage build, the approach validated on the
# InferStation gfx1151 benchmark fleet (halo5 / halo6):
#
#   stage 1 (builder):   git clone AMD's ROCm/vllm `gfx11` branch and build the
#                        HIP extensions for gfx1151, on top of AMD's official
#                        gfx1151 base image.
#   stage 2 (runtime):   install that wheel onto the same base.
#
# Why the ROCm/vllm `gfx11` branch (not upstream vllm-project/vllm main):
#   - it carries gfx1151-specific kernels (W4A16 prefill M-aware BLOCK_N,
#     GDN prefill shape-keyed config, unquantized-weight stride padding off the
#     gfx11x 4096B cliff) that upstream main does not have, and
#   - it keeps FusedMoE.tp_size, so AWQ MoE models load (upstream main hit
#     `RoutedExperts has no attribute tp_size` on ROCm).
#   Tradeoff: it lags upstream main and has no DiffusionGemma support.

# AMD's official gfx1151 base (ROCm 7.13 / PyTorch 2.10 / py3.13). Verified base
# on InferStation. The vLLM preinstalled in the base (0.19.1) is overwritten by
# the wheel built below.
ARG VLLM_BASE=rocm/vllm:rocm7.13.0_gfx1151_ubuntu24.04_py3.13_pytorch_2.10.0_vllm_0.19.1

ARG VLLM_REPO=https://github.com/ROCm/vllm.git
ARG VLLM_TAG=gfx11
ARG PYTORCH_ROCM_ARCH=gfx1151

# Cache-bust for the `git clone` layer. `gfx11` is a moving branch, so without
# this the clone layer would be cached forever and freeze the build at an old
# commit. Pass --build-arg CACHEBUST=$(git ls-remote https://github.com/ROCm/vllm.git gfx11 | cut -f1)
ARG CACHEBUST=gfx11

# =========================================================
# STAGE 1: build the vLLM wheel for gfx1151
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

# Clone the gfx11 branch. CACHEBUST in the command text invalidates this layer
# whenever the branch HEAD changes.
RUN echo "cache-bust: ${CACHEBUST}" && \
    git clone --depth=1 --branch "${VLLM_TAG}" "${VLLM_REPO}" vllm

# gfx1151 build fix: the gfx11 branch uses C++23 `std::in_range` in
# csrc/rocm/skinny_gemms_int4.cu, but HIP device compilation here is C++20, so
# the symbol is unavailable. Rewrite it to an equivalent C++17 bounds check.
# (Same patch carried on InferStation — see mods/fix-gfx11-in-range.)
RUN cd vllm && \
    sed -i 's/std::in_range<int>(\([^)]*\))/(\1 <= static_cast<int64_t>(std::numeric_limits<int>::max()))/g' \
        csrc/rocm/skinny_gemms_int4.cu && \
    sed -i '1i #include <limits>' csrc/rocm/skinny_gemms_int4.cu

# Build the wheel (HIP extensions compiled for gfx1151). Takes ~30-60 min cold,
# faster with ccache warm.
RUN cd vllm && \
    python3 -m pip install --no-cache-dir build setuptools-rust && \
    python3 setup.py bdist_wheel && \
    mkdir -p /wheels && cp dist/*.whl /wheels/

# =========================================================
# STAGE 2: runtime
# =========================================================
FROM ${VLLM_BASE} AS runtime
ARG PYTORCH_ROCM_ARCH
ARG VLLM_TAG

LABEL io.amd.rocm.arch="${PYTORCH_ROCM_ARCH}" \
      org.opencontainers.image.description="vLLM (${VLLM_TAG}) for AMD Strix Halo (${PYTORCH_ROCM_ARCH})"

ENV PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH}
# gfx1151 = RDNA3.5. The official base usually detects it correctly; only set an
# override if ROCm misidentifies the arch on your host.
# ENV HSA_OVERRIDE_GFX_VERSION=11.5.1

# Reinstall vLLM from the wheel just built (overwrites the base's 0.19.1).
COPY --from=builder /wheels/*.whl /tmp/wheels/
RUN python3 -m pip install --no-cache-dir --force-reinstall --no-deps /tmp/wheels/*.whl && \
    rm -rf /tmp/wheels

WORKDIR /workspace

# The launcher clears the entrypoint and runs init before vLLM starts.
CMD ["/bin/bash"]
