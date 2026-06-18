#!/bin/bash
# Example: serve a model on a single Strix Halo node. (SCAFFOLD — placeholder.)
# TODO: fill in a real, hardware-verified launch once launch-cluster.sh works.
set -e

./launch-cluster.sh --solo -p 8000:8000 exec \
  vllm serve "Qwen/Qwen3-8B" \
    --port 8000 --host 0.0.0.0 \
    --gpu-memory-utilization 0.9 \
    --attention-backend TRITON_ATTN
