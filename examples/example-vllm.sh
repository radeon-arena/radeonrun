#!/bin/bash
# Example: serve Qwen3.6-35B-A3B (BF16) on a single Strix Halo node.
# Verified serve command (TRITON_ATTN is the only stable vLLM attention on gfx1151).
set -e

MODELS_DIR=/models ./launch-cluster.sh --solo -p 8000:8000 exec \
  vllm serve /models/Qwen3.6-35B-A3B \
    --host 0.0.0.0 --port 8000 \
    --max-num-seqs 32 --dtype bfloat16 \
    --attention-backend TRITON_ATTN
