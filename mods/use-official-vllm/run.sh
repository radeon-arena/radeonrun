#!/bin/bash
set -euo pipefail
# Mod: use-official-vllm (SCAFFOLD)
#
# Compatibility setup for running an official/third-party vLLM ROCm image with
# this launcher. The upstream (CUDA) version installs git and fixes an NCCL
# soname clash; the ROCm equivalent (RCCL, gfx1151) is TODO and must be
# verified on hardware.
PREFIX="[use-official-vllm]"
echo "$PREFIX SCAFFOLD — not implemented for ROCm yet."
# TODO:
#   - ensure git/pytest present
#   - reconcile pip-installed vs system RCCL if both exist
