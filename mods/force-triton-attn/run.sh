#!/bin/bash
set -euo pipefail
# Mod: force-triton-attn
#
# On gfx1151, FLASH_ATTN is a dead end in vLLM (get_flash_attn_version() returns
# None on ROCm). TRITON_ATTN is the only stable attention backend. This mod is a
# reminder/helper: it exports the env that pins the attention backend so a serve
# command that omits --attention-backend still lands on TRITON_ATTN.
#
# See docs/GFX1151_NOTES.md.

export VLLM_ATTENTION_BACKEND=TRITON_ATTN
echo "[force-triton-attn] VLLM_ATTENTION_BACKEND=TRITON_ATTN (FLASH_ATTN is unsupported on gfx1151)"
