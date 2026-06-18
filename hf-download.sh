#!/bin/bash
set -euo pipefail
#
# hf-download.sh - Download a model from HuggingFace into a local models dir.
#
# This is the download flow used to stage models for the gfx1151 serve runs.
#
# Usage:
#   ./hf-download.sh <org/model>
#   MODELS_DIR=/models ./hf-download.sh Qwen/Qwen3-30B-A3B
#
# Env:
#   MODELS_DIR  destination root (default: /models) -> /models/<repo-basename>
#   HF_TOKEN    HuggingFace token for gated/private repos (optional)

usage() {
    echo "Usage: $0 <org/model>"
    echo "  e.g. $0 Qwen/Qwen3-30B-A3B"
    echo "  Env: MODELS_DIR (default /models), HF_TOKEN (optional)"
}

if [[ $# -lt 1 ]]; then usage; exit 1; fi

MODEL="$1"
MODELS_DIR="${MODELS_DIR:-/models}"
DEST="${MODELS_DIR}/$(basename "$MODEL")"

# Pick the hf CLI (huggingface_hub[cli]); fall back to python -m.
if command -v hf >/dev/null 2>&1; then
  HF=(hf)
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF=(huggingface-cli)
else
  echo "hf CLI not found. Install: pip install --user 'huggingface_hub[cli]'" >&2
  exit 1
fi

# XET backend can flake on parallel pulls of the same repo; disable it.
export HF_HUB_DISABLE_XET=1
[[ -n "${HF_TOKEN:-}" ]] && export HF_TOKEN

mkdir -p "$DEST"
echo "Downloading $MODEL -> $DEST"
"${HF[@]}" download "$MODEL" --local-dir "$DEST"
echo "Done: $DEST"
