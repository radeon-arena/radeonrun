#!/bin/bash
#
# autodiscover.sh - Discover cluster nodes / network interfaces and write .env.
#                   (SCAFFOLD — not implemented.)
#
# TODO:
#   - detect local IP / high-speed interface
#   - probe peer nodes (if multi-node is in scope for Strix Halo)
#   - write CLUSTER_NODES / ETH_IF / IB_IF / LOCAL_IP into .env

SCRIPT_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"

echo "[scaffold] autodiscover.sh is not implemented yet (dir: $SCRIPT_DIR)."
exit 1
