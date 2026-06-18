#!/bin/bash
set -euo pipefail
# Mod: use-official-vllm (SCAFFOLD)
#
# Compatibility setup for running an official/third-party vLLM ROCm image with
# this launcher. Installs git and reconciles an RCCL soname clash on gfx1151.
# TODO and must be verified on hardware.
PREFIX="[use-official-vllm]"
echo "$PREFIX SCAFFOLD — not implemented for ROCm yet."
# TODO:
#   - ensure git/pytest present
#   - reconcile pip-installed vs system RCCL if both exist
