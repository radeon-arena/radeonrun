#!/usr/bin/env bash
set -euo pipefail
#
# build.sh - Build a radeonrun image for a given framework x device.
#
# radeonrun ships one Dockerfile per inference framework; the target GPU
# (and its base image) is selected by a device profile in devices/*.env. This
# script wires the two together.
#
# Usage:
#   ./build.sh --framework vllm      --device halo      # gfx1151 vLLM (gfx11 branch)
#   ./build.sh --framework vllm-main --device halo      # gfx1151 vLLM (upstream main)
#   ./build.sh --framework llamacpp  --device halo      # gfx1151 llama.cpp (HIP)
#   ./build.sh -f llamacpp -d w7900                      # gfx1100 llama.cpp
#   ./build.sh -f vllm -d halo -t myrepo/vllm:test       # custom image tag
#   ./build.sh -f llamacpp -d halo --push                # build, then push to ghcr
#
# Frameworks: vllm | vllm-main | llamacpp
# Devices:    halo (gfx1151) | w7900 (gfx1100) | r9700 (gfx1201)
#
# Image name puts the device first; the tag carries the upstream build commit
# (byte-reproducible), and `:latest` is also tagged for convenience:
#   ghcr.io/radeon-arena/<device>-<image>:<commit>   (+ :latest)
# where <image> = vllm | vllm-main | llamacpp.
# (Strix Halo is single-node here; multi-node copy is intentionally omitted.)

FRAMEWORK="vllm"
DEVICE="halo"
IMAGE_TAG=""
PUSH=0
REGISTRY="${RADEONRUN_IMAGE_REGISTRY:-ghcr.io/radeon-arena}"

docker_cmd=()
if [[ -n "${RADEONRUN_DOCKER:-}" ]]; then
  read -r -a docker_cmd <<<"${RADEONRUN_DOCKER}"
elif docker info >/dev/null 2>&1; then
  docker_cmd=(docker)
elif sudo -n docker info >/dev/null 2>&1; then
  docker_cmd=(sudo -n docker)
else
  echo "No usable Docker command; tried docker and sudo -n docker" >&2
  exit 1
fi

usage() { sed -n '3,25p' "$0"; exit "${1:-0}"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    -f|--framework) FRAMEWORK="$2"; shift 2 ;;
    -d|--device)    DEVICE="$2"; shift 2 ;;
    -t|--tag)       IMAGE_TAG="$2"; shift 2 ;;
    --registry)     REGISTRY="$2"; shift 2 ;;
    -p|--push)      PUSH=1; shift ;;
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
    COMMIT="${CACHEBUST:0:10}"
    BUILD_ARGS=(--build-arg "VLLM_BASE=${VLLM_BASE}" --build-arg "PYTORCH_ROCM_ARCH=${GFX}" --build-arg "CACHEBUST=${CACHEBUST}")
    ;;
  vllm-main)
    DOCKERFILE="dockerfiles/vllm-main/Dockerfile"
    IMAGE="vllm-main"
    : "${VLLM_BASE:?device '$DEVICE' has no VLLM_BASE set (see devices/${DEVICE}.env)}"
    CACHEBUST="$(git ls-remote https://github.com/vllm-project/vllm.git main 2>/dev/null | cut -f1 || true)"
    CACHEBUST="${CACHEBUST:-main-$(date -u +%Y%m%d)}"
    COMMIT="${CACHEBUST:0:10}"
    BUILD_ARGS=(--build-arg "VLLM_BASE=${VLLM_BASE}" --build-arg "PYTORCH_ROCM_ARCH=${GFX}" --build-arg "CACHEBUST=${CACHEBUST}")
    ;;
  llamacpp)
    DOCKERFILE="dockerfiles/llamacpp/Dockerfile"
    IMAGE="llamacpp"
    LLAMA_HEAD="$(git ls-remote https://github.com/ggml-org/llama.cpp.git master 2>/dev/null | cut -f1 || true)"
    LLAMA_HEAD="${LLAMA_HEAD:-master-$(date -u +%Y%m%d)}"
    COMMIT="${LLAMA_HEAD:0:10}"
    BUILD_ARGS=(--build-arg "ROCM_DOCKER_ARCH=${GFX}" --build-arg "ROCM_VERSION=${ROCM_VERSION:-7.2.1}")
    ;;
  *)
    echo "Unknown framework: $FRAMEWORK (want vllm | vllm-main | llamacpp)" >&2; exit 2 ;;
esac

REGISTRY="${REGISTRY%/}"
REPO="${REGISTRY}/${DEVICE}-${IMAGE}"
IMAGE_TAG="${IMAGE_TAG:-${REPO}:${COMMIT}}"

# A custom --tag is a complete OCI reference. Its companion moving tag belongs
# to the same repository; never create an unrelated ghcr.io/radeon-arena tag.
IMAGE_NO_DIGEST="${IMAGE_TAG%@*}"
LAST_COMPONENT="${IMAGE_NO_DIGEST##*/}"
if [[ "$LAST_COMPONENT" == *:* ]]; then
  TARGET_REPO="${IMAGE_NO_DIGEST%:*}"
else
  TARGET_REPO="$IMAGE_NO_DIGEST"
fi
LATEST_TAG="${TARGET_REPO}:latest"

echo "Building ${FRAMEWORK} for ${DEVICE} (${GFX})  ->  ${IMAGE_TAG}  (+ ${LATEST_TAG})"
echo "  dockerfile: ${DOCKERFILE}"
# --network=host: the build's apt-get needs DNS; on hosts using systemd-resolved
# (127.0.0.53 stub) the default bridge build network can't resolve, so share the
# host network namespace for the build.
( cd "$SCRIPT_DIR" && "${docker_cmd[@]}" build --network=host -f "$DOCKERFILE" -t "$IMAGE_TAG" -t "$LATEST_TAG" "${BUILD_ARGS[@]}" . )

echo "Done: ${IMAGE_TAG}"
if [[ "$PUSH" == "1" ]]; then
  echo "Pushing ${IMAGE_TAG} and ${LATEST_TAG} ..."
  if "${docker_cmd[@]}" push "$IMAGE_TAG" && "${docker_cmd[@]}" push "$LATEST_TAG"; then
    echo "Pushed: ${IMAGE_TAG} (+ :latest)"
  else
    echo "WARNING: push failed -- image is built and usable locally, but not synced to ghcr." >&2
    echo "         Check docker login/permissions for the target registry." >&2
  fi
else
  echo "Push with:  docker push ${IMAGE_TAG} && docker push ${LATEST_TAG}"
fi

