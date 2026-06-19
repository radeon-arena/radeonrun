#!/bin/bash
set -euo pipefail
#
# launch-cluster.sh - Run the radeon-docker container in solo mode.
#
# Solo mode is implemented and uses the GPU passthrough verified on the
# InferStation gfx1151 nodes. Cluster (multi-node) mode is NOT implemented —
# Strix Halo is a single-GPU APU and our experience here is single-node only.
#
# Usage:
#   ./launch-cluster.sh --solo exec vllm serve /models/<m> --host 0.0.0.0 --port 8000 ...
#   ./launch-cluster.sh --solo -p 8000:8000 exec vllm serve /models/<m> ...
#
# Environment:
#   IMAGE                container image            (default: ghcr.io/radeon-arena/vllm:gfx1151)
#   CONTAINER            container name             (default: halo_vllm)
#   MODELS_DIR           host models dir -> /models (default: /models)
#   HF_HOME              host HF cache              (default: ~/.cache/huggingface)
#   HIP_VISIBLE_DEVICES  GPU index to expose        (optional)

IMAGE="${IMAGE:-ghcr.io/radeon-arena/vllm:gfx1151}"
CONTAINER="${CONTAINER:-radeon_vllm}"
MODELS_DIR="${MODELS_DIR:-/models}"
HF_CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
PORT_MAP=""
SOLO=0

# AMD ROCm GPU passthrough — verified on InferStation halo nodes.
AMD_DEVICES=(--device /dev/kfd --device /dev/dri --group-add video
             --security-opt seccomp=unconfined --ipc host)

args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --solo) SOLO=1; shift ;;
    -p) PORT_MAP="$2"; shift 2 ;;
    exec) shift; args=("$@"); break ;;
    *) echo "Unknown arg before 'exec': $1" >&2; exit 2 ;;
  esac
done

if [[ "$SOLO" -ne 1 ]]; then
  echo "Only --solo mode is implemented (Strix Halo is single-node). Pass --solo." >&2
  exit 1
fi
if [[ "${#args[@]}" -eq 0 ]]; then
  echo "No command after 'exec'. Example:" >&2
  echo "  ./launch-cluster.sh --solo -p 8000:8000 exec vllm serve /models/m --host 0.0.0.0 --port 8000" >&2
  exit 2
fi

run=(docker run --rm -it --name "$CONTAINER" "${AMD_DEVICES[@]}"
     -v "$MODELS_DIR:/models" -v "$HF_CACHE_DIR:/root/.cache/huggingface")

[[ -n "$PORT_MAP" ]] && run+=(-p "$PORT_MAP")
[[ -n "${HIP_VISIBLE_DEVICES:-}" ]] && run+=(-e "HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES}")

# Clear the image entrypoint so init can run before the server starts.
run+=(--entrypoint /bin/bash "$IMAGE" -lc "exec ${args[*]}")

echo "+ ${run[*]}"
exec "${run[@]}"
