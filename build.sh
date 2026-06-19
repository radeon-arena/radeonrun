#!/usr/bin/env bash
set -euo pipefail
#
# build.sh - Build a radeon-docker image for a given framework x device.
#
# radeon-docker ships one Dockerfile per inference framework; the target GPU
# (and its base image) is selected by a device profile in devices/*.env. This
# script wires the two together.
#
# Usage:
#   ./build.sh --framework vllm      --device strix     # gfx1151 vLLM (gfx11 branch)
#   ./build.sh --framework vllm-main --device strix     # gfx1151 vLLM (upstream main)
#   ./build.sh --framework llamacpp  --device strix     # gfx1151 llama.cpp (HIP)
#   ./build.sh -f llamacpp -d w7900                      # gfx1100 llama.cpp
#   ./build.sh -f vllm -d strix -t myrepo/vllm:test      # custom image tag
#
# Frameworks: vllm | vllm-main | llamacpp
# Devices:    strix (gfx1151) | w7900 (gfx1100) | r9700 (gfx1200)
#
# Default image tag: ghcr.io/radeon-arena/<framework>:<gfx-arch>
# (Strix Halo is single-node here; multi-node copy is intentionally omitted.)

FRAMEWORK="vllm"
DEVICE="strix"
IMAGE_TAG=""
ORG="ghcr.io/radeon-arena"

usage() { sed -n '3,22p' "$0"; exit "${1:-0}"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    -f|--framework) FRAMEWORK="$2"; shift 2 ;;
    -d|--device)    DEVICE="$2"; shift 2 ;;
    -t|--tag)       IMAGE_TAG="$2"; shift 2 ;;
    -h|--help)      usage 0 ;;
    *) echo "Unknown arg: $1" >&2; usage 2 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/devices/${DEVICE}.env"
if [[ ! -f "$ENV_FILE" ]]; then
  echo "No device profile: $ENV_FILE" >&2
  echo "Available: $(cd "${SCRIPT_DIR}/devices" && ls *.env 2>/dev/null | sed 's/\.env//' | tr '\n' ' ')" >&2
  exit 2
fi
# shellcheck disable=SC1090
source "$ENV_FILE"
: "${GFX:?device profile must set GFX}"

case "$FRAMEWORK" in
  vllm)
    DOCKERFILE="dockerfiles/vllm/Dockerfile"
    IMAGE="vllm"
    : "${VLLM_BASE:?device '$DEVICE' has no VLLM_BASE set (see devices/${DEVICE}.env)}"
    CACHEBUST="$(git ls-remote https://github.com/ROCm/vllm.git gfx11 2>/dev/null | cut -f1 || true)"
    CACHEBUST="${CACHEBUST:-gfx11-$(date -u +%Y%m%d)}"
    BUILD_ARGS=(--build-arg "VLLM_BASE=${VLLM_BASE}" --build-arg "PYTORCH_ROCM_ARCH=${GFX}" --build-arg "CACHEBUST=${CACHEBUST}")
    ;;
  vllm-main)
    DOCKERFILE="dockerfiles/vllm-main/Dockerfile"
    IMAGE="vllm-main"
    : "${VLLM_BASE:?device '$DEVICE' has no VLLM_BASE set (see devices/${DEVICE}.env)}"
    CACHEBUST="$(git ls-remote https://github.com/vllm-project/vllm.git main 2>/dev/null | cut -f1 || true)"
    CACHEBUST="${CACHEBUST:-main-$(date -u +%Y%m%d)}"
    BUILD_ARGS=(--build-arg "VLLM_BASE=${VLLM_BASE}" --build-arg "PYTORCH_ROCM_ARCH=${GFX}" --build-arg "CACHEBUST=${CACHEBUST}")
    ;;
  llamacpp)
    DOCKERFILE="dockerfiles/llamacpp/Dockerfile"
    IMAGE="llamacpp"
    BUILD_ARGS=(--build-arg "ROCM_DOCKER_ARCH=${GFX}" --build-arg "ROCM_VERSION=${ROCM_VERSION:-7.2.1}")
    ;;
  *)
    echo "Unknown framework: $FRAMEWORK (want vllm | vllm-main | llamacpp)" >&2; exit 2 ;;
esac

IMAGE_TAG="${IMAGE_TAG:-${ORG}/${IMAGE}:${GFX}}"

echo "Building ${FRAMEWORK} for ${DEVICE} (${GFX})  ->  ${IMAGE_TAG}"
echo "  dockerfile: ${DOCKERFILE}"
( cd "$SCRIPT_DIR" && docker build -f "$DOCKERFILE" -t "$IMAGE_TAG" "${BUILD_ARGS[@]}" . )

echo "Done: ${IMAGE_TAG}"
echo "Push with:  docker push ${IMAGE_TAG}"
