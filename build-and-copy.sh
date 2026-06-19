#!/bin/bash
set -euo pipefail
#
# build-and-copy.sh - Build the halo-vllm-docker image(s) for gfx1151.
#
# Builds the vLLM image (./Dockerfile) by default, pinning the gfx11 branch HEAD
# into CACHEBUST so the build tracks AMD's gfx1151 fork (see Dockerfile).
#
# Usage:
#   ./build-and-copy.sh                 # vLLM gfx11 image  -> halo-vllm
#   ./build-and-copy.sh --main          # vLLM upstream-main -> halo-vllm-main (DiffusionGemma)
#   ./build-and-copy.sh --llamacpp      # llama.cpp HIP      -> halo-llamacpp
#   ./build-and-copy.sh -t mytag        # custom image tag
#
# (Multi-node copy is not implemented — Strix Halo is single-node here.)

IMAGE_TAG=""
ENGINE="vllm"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --llamacpp) ENGINE="llamacpp"; shift ;;
    --main)     ENGINE="vllm-main"; shift ;;
    -t|--tag)   IMAGE_TAG="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ "$ENGINE" == "llamacpp" ]]; then
  IMAGE_TAG="${IMAGE_TAG:-halo-llamacpp}"
  echo "Building llama.cpp (HIP/gfx1151) image: $IMAGE_TAG"
  docker build -f Dockerfile.llamacpp -t "$IMAGE_TAG" .
elif [[ "$ENGINE" == "vllm-main" ]]; then
  IMAGE_TAG="${IMAGE_TAG:-halo-vllm-main}"
  # Track upstream vllm-project/vllm main HEAD (has DiffusionGemma).
  CACHEBUST="$(git ls-remote https://github.com/vllm-project/vllm.git main 2>/dev/null | cut -f1 || true)"
  CACHEBUST="${CACHEBUST:-main-$(date -u +%Y%m%d)}"
  echo "Building vLLM (upstream main, gfx1151) image: $IMAGE_TAG  (CACHEBUST=$CACHEBUST)"
  docker build -f Dockerfile.main -t "$IMAGE_TAG" --build-arg "CACHEBUST=$CACHEBUST" .
else
  IMAGE_TAG="${IMAGE_TAG:-halo-vllm}"
  # Resolve the current gfx11 branch HEAD so the wheel build tracks it (the
  # Dockerfile's git-clone layer is otherwise cached forever).
  CACHEBUST="$(git ls-remote https://github.com/ROCm/vllm.git gfx11 2>/dev/null | cut -f1 || true)"
  CACHEBUST="${CACHEBUST:-gfx11-$(date -u +%Y%m%d)}"
  echo "Building vLLM (gfx1151) image: $IMAGE_TAG  (CACHEBUST=$CACHEBUST)"
  docker build -f Dockerfile -t "$IMAGE_TAG" --build-arg "CACHEBUST=$CACHEBUST" .
fi

echo "Done: $IMAGE_TAG"
