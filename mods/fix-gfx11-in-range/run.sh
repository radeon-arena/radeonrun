#!/bin/bash
set -euo pipefail
# Mod: fix-gfx11-in-range
#
# AMD's ROCm/vllm `gfx11` branch uses C++23 `std::in_range` in
# csrc/rocm/skinny_gemms_int4.cu, but HIP device compilation on gfx1151 runs at
# C++20, so the symbol is missing and the build fails:
#   error: no member named 'in_range' in namespace 'std'
#
# This rewrites it to an equivalent C++17 bounds check. Run from the vLLM source
# tree (the Dockerfile applies the same sed inline at build time).
#
# Usage: cd /path/to/vllm && /path/to/mods/fix-gfx11-in-range/run.sh

TARGET="csrc/rocm/skinny_gemms_int4.cu"
if [[ ! -f "$TARGET" ]]; then
  echo "[fix-gfx11-in-range] $TARGET not found; run from the vLLM source root." >&2
  exit 1
fi

# std::in_range<int>(x)  ->  (x <= static_cast<int64_t>(std::numeric_limits<int>::max()))
sed -i 's/std::in_range<int>(\([^)]*\))/(\1 <= static_cast<int64_t>(std::numeric_limits<int>::max()))/g' "$TARGET"
# ensure <limits> is included
grep -q '#include <limits>' "$TARGET" || sed -i '1i #include <limits>' "$TARGET"

echo "[fix-gfx11-in-range] patched $TARGET (C++23 std::in_range -> C++17 bounds check)"
