# Strix Halo (gfx1151) vLLM notes

Hard-won facts from running vLLM and llama.cpp on AMD Strix Halo
(Radeon 8060S iGPU, `gfx1151`, RDNA 3.5) on the InferStation benchmark fleet.
These are the things that cost real debugging time — read before you fight the
same battles.

## Hardware / driver

- `gfx1151` is RDNA 3.5 (an APU iGPU), compute capability 11.5. The box has
  **128 GB unified memory** shared between CPU and GPU, so VRAM is large but
  memory bandwidth (LPDDR5X, ~256 GB/s) is the decode bottleneck.
- **ROCm version must match the host KFD driver.** Our halo nodes run ROCm
  **7.2.1**; `gfx1151` is **not** supported on ROCm 7.0. The llama.cpp image base
  is pinned to `rocm/dev-ubuntu-24.04:7.2.1-complete` for this reason.

## vLLM: which image / branch

- Base: `rocm/vllm:rocm7.13.0_gfx1151_ubuntu24.04_py3.13_pytorch_2.10.0_vllm_0.19.1`
  (AMD's official gfx1151 base; the bundled vLLM 0.19.1 gets overwritten).
- Build vLLM from **AMD's `ROCm/vllm` `gfx11` branch**, not upstream
  `vllm-project/vllm` main, because:
  - `gfx11` carries gfx1151-specific kernels (W4A16 prefill M-aware `BLOCK_N`,
    GDN prefill shape-keyed config, unquantized-weight stride padding off the
    gfx11x 4096B cliff) that upstream main lacks.
  - `gfx11` keeps `FusedMoE.tp_size`. Upstream main refactored MoE into
    `RoutedExperts` and its `moe_wna16_weight_loader` still reads `tp_size`,
    so **AWQ MoE models crash on load on ROCm** with
    `AttributeError: 'RoutedExperts' object has no attribute 'tp_size'`.
  - Tradeoff: `gfx11` lags upstream main and has **no DiffusionGemma**. If you
    need DiffusionGemma, build a separate image from upstream main and pin it.

### Build gotcha: C++23 `std::in_range`

The `gfx11` branch uses `std::in_range` (C++23) in
`csrc/rocm/skinny_gemms_int4.cu`, but HIP **device** compilation runs at C++20,
so the symbol is missing and the build fails with
`error: no member named 'in_range' in namespace 'std'`.

- Changing `CMAKE_HIP_STANDARD` to 23 does **not** help (the `-std` on the HIP
  device line comes from elsewhere).
- Fix = patch the source to a C++17-equivalent bounds check (see
  [`mods/fix-gfx11-in-range`](../mods/fix-gfx11-in-range/) and the `sed` in the
  Dockerfile):
  `std::in_range<int>(x)` → `(x <= static_cast<int64_t>(std::numeric_limits<int>::max()))`
  plus `#include <limits>`.

## vLLM: attention backends on gfx1151

| Backend | Status on gfx1151 |
|---|---|
| `TRITON_ATTN` | ✅ **stable — use this** |
| `ROCM_ATTN` | ✅ works |
| `FLASH_ATTN` | ❌ **dead end** (see below) |
| `ROCM_AITER_FA` | ❌ invalid — requires CDNA/MI (`on_mi3xx()`) |

**FLASH_ATTN does not work on gfx1151 in vLLM**, and it is not a config you can
fix:

- `get_flash_attn_version()` is hardcoded `if current_platform.is_rocm(): return None`.
- Downstream that yields, for real models:
  - Qwen3.6-35B-A3B + FLASH_ATTN → `AssertionError: FlashAttention version not detected.`
  - Gemma-4 + FLASH_ATTN → `ValueError: ... FLASH_ATTN ... Reason: ['head_size not supported']`
- This is a vLLM-on-ROCm design limit, not missing packages. The Triton-AMD
  flash bits are present (`flash_attn 2.8.3`, `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`)
  but vLLM still won't use them. **Only TRITON_ATTN is the stable path.**

> Note: this is the opposite of DGX Spark / CUDA, where FLASH_ATTN works and
> TRITON is the fallback. Don't copy CUDA attention choices to Halo.

## vLLM: MoE / quantization — no marlin on ROCm

On ROCm, vLLM disables marlin for **every** quant's MoE path; experts fall back
to Triton `fused_experts`. This is by design, not a bug, and **cannot be
"fixed" into marlin**:

- `utils/marlin_utils.py check_moe_marlin_supports_layer` first line:
  `if current_platform.is_rocm(): return False`.
- The ROCm build ships **no marlin MoE C-ops** (`moe_wna16_marlin_gemm` etc.
  are absent). Patching the guard just moves the crash to
  `NotImplementedError: No WNA16 MoE backend supports the deployment configuration`.
- So on Halo: **dense** layers may use awq_marlin via the Triton path, but **MoE
  experts always run Triton WNA16 `fused_experts`** (AWQ/GPTQ/Quark all land
  here; BF16 MoE uses the unquantized Triton fused MoE). marlin = NVIDIA
  tensor-core PTX, never ported to RDNA3.5.

## vLLM: AITER is off here

AMD's AITER acceleration library is **not installed** in the gfx1151 image, and
most AITER paths are gated on `on_mi3xx()` (CDNA/MI) anyway, so `gfx1151` would
not use them even if present. The master switch `VLLM_ROCM_USE_AITER` defaults
to false. Don't expect AITER MHA/MLA/MoE on Halo.

## vLLM: what AMD optimizations *do* apply

- **Skinny GEMM** HIP kernels (`_rocm_C`) for dense GEMMs, gated on `on_gfx1x()`
  which gfx1151 hits (`VLLM_ROCM_USE_SKINNY_GEMM` defaults on):
  - single-token decode (`n==1`, `k<=8192`) → `ops.LLMM1`
  - small-batch decode (`0 < n <= 4`) → `ops.wvSplitK`
  - prefill / large batch → rocBLAS / hipBLASLt
  - the more aggressive `wvSplitKrc`/`wvSplitKQ` are `on_gfx950()` (MI350) only.
- MoE experts: Triton `fused_experts` with `VLLM_ROCM_MOE_PADDING=True`.
- attention: TRITON_ATTN (see above).

## vLLM: health 500 (FastAPI regression)

If the engine starts (`Application startup complete`) but `/health` returns
**500**, you likely hit a FastAPI ≥ 0.137 + `prometheus-fastapi-instrumentator`
≤ 8.0.0 clash: a lazy `_IncludedRouter` has no `.path`, and the metrics
middleware does `route.path` on every request → `AttributeError` → 500 on
`/health`. Fixed in current vLLM via `_patch_instrumentator_route_walk`
(skips routes with no `.path`). Check the image with
`grep _patch_instrumentator_route_walk .../instrumentator/metrics.py`.

## Running: GPU passthrough

```bash
docker run --rm -it \
  --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --ipc host \
  -v /models:/models \
  ghcr.io/radeon-arena/vllm:gfx1151 \
  vllm serve /models/<model> --host 0.0.0.0 --port 8000 --attention-backend TRITON_ATTN
```

(`./launch-cluster.sh --solo` wraps exactly this.)

## llama.cpp on gfx1151

- Build with `-DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1151` on the ROCm 7.2.1 base.
- Serve: `llama-server -m <gguf> -ngl 999 -fa on --host 0.0.0.0 --port <p> -c <ctx> -np <n>`.
  Unlike vLLM, llama.cpp's `-fa on` (flash attention) **works** on gfx1151.
- `-c` (context) is **shared across all `-np` slots**; each slot gets `ctx/np`.
  Every slot must fit `input + output` tokens, or requests 400 with
  `exceeds available context size`. Size `-c >= np * (in + out) * margin`.
- Some quants need `-fa off` (e.g. MiMo-V2.5 UD-Q2_K_XL ran with `-fa off --no-mmap`).
- For GGUF weights, llama.cpp HIP is frequently the most efficient engine on
  Halo (vLLM BF16 often runs ~30-40% of llama.cpp's tok/s because the ROCm
  kernels aren't as tuned).
