# Mods and Patches

A **mod** is a directory with a `run.sh` (and optional patch/template files)
that applies a specific fix — at image build time (source patches) or at launch
time (env/runtime tweaks).

## Convention

```
mods/<mod-name>/
  run.sh            # the fix; build-time mods run in the vLLM source tree
  *.patch / *.jinja # optional supporting files
```

A recipe references mods by path:

```yaml
mods:
  - mods/force-triton-attn
```

## Current mods (from real gfx1151 experience)

- **`fix-gfx11-in-range/`** — build fix. AMD's `ROCm/vllm` `gfx11` branch uses
  C++23 `std::in_range` in `csrc/rocm/skinny_gemms_int4.cu`, which fails under
  HIP's C++20 device compile. Rewrites it to a C++17 bounds check. The
  Dockerfile applies the same patch inline. See [docs/GFX1151_NOTES.md](../docs/GFX1151_NOTES.md).

- **`force-triton-attn/`** — runtime helper. Pins `VLLM_ATTENTION_BACKEND=TRITON_ATTN`
  because FLASH_ATTN is unsupported on gfx1151 in vLLM.
