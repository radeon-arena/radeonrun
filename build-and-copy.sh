#!/bin/bash
set -e
#
# build-and-copy.sh - Build the halo-vllm-docker image and optionally copy it
#                     to other nodes. (SCAFFOLD — not implemented.)
#
# Mirrors the upstream flags so the real build can be filled in later.
#
# Usage (intended):
#   ./build-and-copy.sh                 # build locally (single Strix Halo node)
#   ./build-and-copy.sh -c              # build + copy to COPY_HOSTS in .env
#   ./build-and-copy.sh --rebuild-vllm  # build vLLM from source instead of wheel
#
# TODO:
#   - docker build with the ROCm Dockerfile (ARG ROCM_IMAGE / PYTORCH_ROCM_ARCH)
#   - tag handling (vllm-node, variants)
#   - optional `docker save | ssh host docker load` distribution
#   - record build-metadata.yaml

IMAGE_TAG="halo-vllm-node"

echo "[scaffold] build-and-copy.sh is not implemented yet."
echo "[scaffold] would build image '$IMAGE_TAG' from ./Dockerfile (ROCm/gfx1151)."
exit 1
