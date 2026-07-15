#!/bin/bash
set -euo pipefail
#
# launch-cluster.sh - Run the radeonrun container in solo mode.
#
# Solo means one host and supports one or more locally visible Radeon GPUs.
# Cluster (multi-node) mode is not implemented. GPU count / tensor parallelism
# is controlled by the launch environment and serve command.
#
# Usage:
#   ./launch-cluster.sh --solo exec vllm serve /models/<m> --host 0.0.0.0 --port 8000 ...
#   ./launch-cluster.sh --solo -p 8000:8000 exec vllm serve /models/<m> ...
#
# Environment:
#   IMAGE                container image            (default: ghcr.io/radeon-arena/halo-vllm:latest)
#   CONTAINER            container name             (default: halo_vllm)
#   MODELS_DIR           host models dir -> /models (default: /models)
#   HF_HOME              host HF cache              (default: ~/.cache/huggingface)
#   HIP_VISIBLE_DEVICES  GPU index to expose        (optional)
#   ROCR_VISIBLE_DEVICES GPU index to expose        (optional)
#   RADEONRUN_DOCKER     docker command override    (optional, e.g. "sudo -n docker")

IMAGE="${IMAGE:-ghcr.io/radeon-arena/halo-vllm:latest}"
CONTAINER="${CONTAINER:-radeon_vllm}"
MODELS_DIR="${MODELS_DIR:-/models}"
HF_CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
PORT_MAP=""
SOLO=0

docker_cmd=()
if [[ -n "${RADEONRUN_DOCKER:-}" ]]; then
  read -r -a docker_cmd <<<"${RADEONRUN_DOCKER}"
elif docker info >/dev/null 2>&1; then
  docker_cmd=(docker)
elif sudo -n docker info >/dev/null 2>&1; then
  docker_cmd=(sudo -n docker)
else
  echo "No usable Docker command; tried docker and sudo -n docker" >&2
  exit 1
fi

# AMD ROCm GPU passthrough. Numeric GIDs work even when minimal images do not
# define video/render group names.
AMD_DEVICES=(--device /dev/kfd --device /dev/dri
             --security-opt seccomp=unconfined --ipc host)
for group in video render; do
  gid="$(getent group "$group" | cut -d: -f3 || true)"
  [[ -n "$gid" ]] && AMD_DEVICES+=(--group-add "$gid")
done

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

# Attach a TTY only when we actually have one. CI / background runs are not
# interactive, and `docker run -t` fails there with
# "cannot attach stdin to a TTY-enabled container because stdin is not a terminal".
DOCKER_TTY=""; [ -t 1 ] && DOCKER_TTY="-it"
# Remove any leftover container with this name first. `docker run --rm` only
# removes a container when it EXITS, so a previous run's detached server can
# still be alive holding this name + the port; without this the new run would
# fail to start and the benchmark would hit the STALE model.
"${docker_cmd[@]}" rm -f "$CONTAINER" >/dev/null 2>&1 || true
run=("${docker_cmd[@]}" run --rm $DOCKER_TTY --name "$CONTAINER" "${AMD_DEVICES[@]}"
     -v "$MODELS_DIR:/models" -v "$HF_CACHE_DIR:/root/.cache/huggingface")

[[ -n "$PORT_MAP" ]] && run+=(-p "$PORT_MAP")
[[ -n "${HIP_VISIBLE_DEVICES:-}" ]] && run+=(-e "HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES}")
[[ -n "${ROCR_VISIBLE_DEVICES:-}" ]] && run+=(-e "ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES}")

# Clear the image entrypoint so init can run before the server starts. Quote each
# argument before handing it to `bash -lc`; recipe commands include JSON strings
# (for example vLLM diffusion flags) whose quotes must survive the extra shell.
cmd="exec"
for a in "${args[@]}"; do
  printf -v q ' %q' "$a"
  cmd+="$q"
done
run+=(--entrypoint /bin/bash "$IMAGE" -lc "$cmd")

echo "+ ${run[*]}"
exec "${run[@]}"
