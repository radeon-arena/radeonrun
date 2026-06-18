#!/bin/bash
set -e
#
# hf-download.sh - Download a model from HuggingFace and optionally copy it to
#                  other nodes. (SCAFFOLD — not implemented.)
#
# Usage (intended):
#   ./hf-download.sh <org/model>
#   ./hf-download.sh <org/model> -c --copy-parallel
#
# TODO:
#   - use `hf download` (huggingface_hub[cli]) into the HF cache
#   - optional rsync/ssh distribution to COPY_HOSTS
#   - HF_HUB_DISABLE_XET / token handling

usage() {
    echo "Usage: $0 [OPTIONS] <model-name>"
    echo "  <model-name>            HuggingFace model id (e.g. 'Qwen/Qwen3-8B')"
    echo "  -c, --copy-to <hosts>   Host(s) to copy the model to"
    echo "      --copy-parallel     Copy to all hosts in parallel"
    echo "  -u, --user <user>       Username for ssh (default: \$USER)"
}

if [[ $# -eq 0 ]]; then usage; exit 1; fi

echo "[scaffold] hf-download.sh is not implemented yet."
echo "[scaffold] would download: $*"
exit 1
