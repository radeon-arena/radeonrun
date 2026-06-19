#!/bin/bash
set -euo pipefail
#
# build-and-copy.sh - DEPRECATED back-compat shim.
#
# Superseded by ./build.sh, which builds any framework x device. This wrapper
# keeps the old flags working (it always targets the strix / gfx1151 device):
#
#   ./build-and-copy.sh              ->  ./build.sh -f vllm      -d strix
#   ./build-and-copy.sh --main       ->  ./build.sh -f vllm-main -d strix
#   ./build-and-copy.sh --llamacpp   ->  ./build.sh -f llamacpp  -d strix
#   ./build-and-copy.sh -t mytag     ->  ./build.sh ... -t mytag
#
# Prefer build.sh directly:  ./build.sh --framework llamacpp --device w7900

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK="vllm"
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --llamacpp) FRAMEWORK="llamacpp"; shift ;;
    --main)     FRAMEWORK="vllm-main"; shift ;;
    -t|--tag)   EXTRA+=(--tag "$2"); shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

echo "[build-and-copy.sh] deprecated -> ./build.sh -f ${FRAMEWORK} -d strix"
exec "${SCRIPT_DIR}/build.sh" --framework "${FRAMEWORK}" --device strix "${EXTRA[@]}"
