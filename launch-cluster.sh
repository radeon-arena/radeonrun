#!/bin/bash
#
# launch-cluster.sh - Run the halo-vllm-docker container, solo or cluster.
#                     (SCAFFOLD — not implemented.)
#
# Mirrors the upstream launcher interface so the real implementation can be
# dropped in later.
#
# Usage (intended):
#   ./launch-cluster.sh --solo exec vllm serve <model> --port 8000 --host 0.0.0.0
#   ./launch-cluster.sh --solo -p 8000:8000 exec vllm serve <model> ...
#   ./launch-cluster.sh exec vllm serve <model> -tp 2 --distributed-executor-backend ray
#
# TODO:
#   - parse --solo / -p / exec / mods
#   - docker run with ROCm device passthrough:
#       --device /dev/kfd --device /dev/dri --group-add video
#       --security-opt seccomp=unconfined --ipc host
#   - mount HF cache, clear image entrypoint, run init then the exec command
#   - cluster path (ray head/workers) — TODO, may be out of scope for Halo APU

IMAGE_NAME="halo-vllm-node"
DEFAULT_CONTAINER_NAME="halo_vllm_node"
HF_CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
CONTAINER_WORKSPACE_DIR="/workspace"

echo "[scaffold] launch-cluster.sh is not implemented yet."
echo "[scaffold] image=$IMAGE_NAME container=$DEFAULT_CONTAINER_NAME hf_cache=$HF_CACHE_DIR"
echo "[scaffold] would docker run with ROCm device passthrough and exec: $*"
exit 1
