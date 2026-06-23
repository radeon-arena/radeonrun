# Reproduction — independent verification of every recipe

Every `recipes/*.yaml` in this repo was independently re-run on real AMD
Strix Halo (gfx1151) hardware and the measured numbers committed as native
result files under [`results/strix/`](../results/strix/) (the same schema the
site ingests). This page is the human-readable summary; the JSON files are the
source of truth for the numbers.

> **Two separate projects — do not conflate:**
> - **radeonrun** (this repo) — recipes + container images + the benchmark
>   harness (`bench.py` / `benchmarking/*.yaml`) that **produces** the numbers.
>   The `results/strix/` JSON committed here is the source of truth.
> - **Radeon Arena** — the public website that *displays* the results. It is a
>   separate project and is only a presentation layer; no measurement is attributed
>   to it.

## Method

- Serve command taken verbatim from each recipe (filled with its `defaults`).
- Workload = the repo's own **`benchmarking/halo-arena-v1.yaml`** profile:
  512-in / 128-out streaming, concurrency 1 / 4 / 16 / 32, one warm-up.
- Anchor for "reproduced" = **C=1 single-stream TPOT** (unambiguous, scheduler-
  independent). High-concurrency throughput is reported but treated as
  version/kernel-dependent, not a 1:1 match.
- Each result file carries a `meta.reproduction` block (independent flag, the
  actual quantized weights used where the recipe's are not public, the gfx1151
  host, and a one-line verdict).

## ⚠️ Note on recipe headline numbers

`metadata.measured.decode_toks_per_s` in the recipe YAMLs is a **templated
placeholder**, not a per-recipe measurement — e.g. every Qwen3 Q4_K_M recipe
(4B / 8B / 14B / 32B / 30B-A3B) carries the identical `191.7`, which is
physically impossible. The verdicts below are based on this repo's own
`results/strix/` measurements (the real per-concurrency numbers), **not** the
templated value.

## Results (39 recipes)

| Recipe | Engine | C=1 (decode/TPOT) | C=32 decode | Verdict | JSON |
|---|---|---|---|---|---|
| diffusiongemma-26b-a4b-awq-int4 | vLLM | 6.69 tok/s | 12.29 | ✅ 复现成功(可运行 + 性能与 BF16 同量级) ⚠️非同权重 | [json](../results/strix/diffusiongemma-26b-a4b-awq-int4.json) |
| diffusiongemma-26b-a4b-bf16 | vLLM | 7.93 tok/s | 13.18 | ✅ 复现成功(decode ≈ recipe best) | [json](../results/strix/diffusiongemma-26b-a4b-bf16.json) |
| gemma-4-26b-a4b-it-awq-4bit-vllm | vLLM | 23.16/42.52ms | 236.91 | ✅ 可运行 + 性能合理;⚠️ 非 1:1 同权重对照(详见下) ⚠️非同权重 | [json](../results/strix/gemma-4-26b-a4b-it-awq-4bit-vllm.json) |
| gemma-4-26b-a4b-it-bf16-vllm | vLLM | 22.91/42.97ms | 231.21 | ✅ 复现成功(C=1 单流锚点稳定;高并发吞吐随 batch 提升) | [json](../results/strix/gemma-4-26b-a4b-it-bf16-vllm.json) |
| gemma-4-31b-it-quark-w8a8-int8-vllm | vLLM | 3.08/324.52ms | 52.96 | ✅ 复现成功(C=32 聚合远超 recipe 标称) | [json](../results/strix/gemma-4-31b-it-quark-w8a8-int8-vllm.json) |
| qwen3-30b-a3b-bf16-vllm | vLLM | 14.46/68.27ms | 253.21 | ✅ 复现成功(C=1 单流锚点稳定;高并发吞吐随 batch 提升,如实标注) | [json](../results/strix/qwen3-30b-a3b-bf16-vllm.json) |
| qwen3.6-27b-bf16-vllm | vLLM | 4.3/226.73ms | 51.57 | ✅ 复现成功(逐并发 1:1 同数匹配) | [json](../results/strix/qwen3.6-27b-bf16-vllm.json) |
| qwen3.6-27b-quark-w8a8-int8-vllm | vLLM | 4.05/238.92ms | 48.51 | ✅ 可运行(quark INT8 kernel 成功加载);⚠️ decode 未达 recipe 标称(gfx1151 上 INT8 无加速) | [json](../results/strix/qwen3.6-27b-quark-w8a8-int8-vllm.json) |
| qwen3.6-35b-a3b-awq-4bit-vllm | vLLM | 24.37/38.08ms | 138.17 | ✅ 可运行 + 性能合理;⚠️ 非 1:1 同权重对照 ⚠️非同权重 | [json](../results/strix/qwen3.6-35b-a3b-awq-4bit-vllm.json) |
| qwen3.6-35b-a3b-bf16-vllm | vLLM | 14.76/64.11ms | 122.01 | ✅ 复现成功 | [json](../results/strix/qwen3.6-35b-a3b-bf16-vllm.json) |
| qwen3.6-35b-a3b-quark-w8a8-int8-vllm | vLLM | 17.58/54.42ms | 149.46 | ✅ 复现成功(C=32 聚合超 recipe 标称) | [json](../results/strix/qwen3.6-35b-a3b-quark-w8a8-int8-vllm.json) |
| gemma-4-26b-a4b-it-bf16-llamacpp | llama.cpp | 21.8/45.15ms | 159.5 | ✅ 复现成功(单流 TPOT 锰点稳定;高并发吞吐更高,归因新版 llama.cpp) | [json](../results/strix/gemma-4-26b-a4b-it-bf16-llamacpp.json) |
| gemma-4-26b-a4b-it-q8-0-llamacpp | llama.cpp | 39.72/24.84ms | 188.53 | ✅ 复现成功(单流 TPOT 合理;该模型 C=1 早期一条为异常值,见说明) | [json](../results/strix/gemma-4-26b-a4b-it-q8-0-llamacpp.json) |
| gemma-4-26b-a4b-it-ud-q4-k-m-llamacpp | llama.cpp | 43.56/22.68ms | 221.99 | ✅ 复现成功(单流 TPOT 锰点稳定;高并发吞吐更高,归因新版 llama.cpp) | [json](../results/strix/gemma-4-26b-a4b-it-ud-q4-k-m-llamacpp.json) |
| llama-3.1-8b-instruct-bf16-llamacpp | llama.cpp | 12.2/81.96ms | 121.96 | ✅ 复现成功(单流 TPOT 锰点稳定;高并发吞吐更高,归因新版 llama.cpp) | [json](../results/strix/llama-3.1-8b-instruct-bf16-llamacpp.json) |
| llama-3.1-8b-instruct-q4-k-m-llamacpp | llama.cpp | 41.33/24.16ms | 259.05 | ✅ 复现成功(单流 TPOT 锰点稳定;高并发吞吐更高,归因新版 llama.cpp) | [json](../results/strix/llama-3.1-8b-instruct-q4-k-m-llamacpp.json) |
| mimo-v2.5-ud-q2-k-xl-llamacpp | llama.cpp | 22.3/44.75ms | — | ✅ 复现成功(需特殊内存配置,见下) | [json](../results/strix/mimo-v2.5-ud-q2-k-xl-llamacpp.json) |
| qwen3-14b-bf16-llamacpp | llama.cpp | 7.73/129.38ms | 98.73 | ✅ 复现成功(单流 TPOT 锰点稳定;高并发吞吐更高,归因新版 llama.cpp) | [json](../results/strix/qwen3-14b-bf16-llamacpp.json) |
| qwen3-14b-q4-k-m-llamacpp | llama.cpp | 23.63/42.3ms | 147.9 | ✅ 复现成功(单流 TPOT 锰点稳定;高并发吞吐更高,归因新版 llama.cpp) | [json](../results/strix/qwen3-14b-q4-k-m-llamacpp.json) |
| qwen3-14b-q8-0-llamacpp | llama.cpp | 14.38/69.49ms | 116.25 | ✅ 复现成功(单流 TPOT 锰点稳定;高并发吞吐更高,归因新版 llama.cpp) | [json](../results/strix/qwen3-14b-q8-0-llamacpp.json) |
| qwen3-30b-a3b-bf16-llamacpp | llama.cpp | 25.26/39.55ms | 103.04 | ✅ 复现成功(单流 TPOT 锰点稳定;高并发吞吐更高,归因新版 llama.cpp) | [json](../results/strix/qwen3-30b-a3b-bf16-llamacpp.json) |
| qwen3-30b-a3b-q4-k-m-llamacpp | llama.cpp | 68.4/14.59ms | 296.28 | ✅ 复现成功(单流 TPOT 锰点稳定;高并发吞吐更高,归因新版 llama.cpp) | [json](../results/strix/qwen3-30b-a3b-q4-k-m-llamacpp.json) |
| qwen3-30b-a3b-q8-0-llamacpp | llama.cpp | 50.67/19.7ms | 248.85 | ✅ 复现成功(单流 TPOT 锰点稳定;高并发吞吐更高,归因新版 llama.cpp) | [json](../results/strix/qwen3-30b-a3b-q8-0-llamacpp.json) |
| qwen3-32b-bf16-llamacpp | llama.cpp | 3.37/296.99ms | 42.42 | ✅ 复现成功(单流 TPOT 锰点稳定;高并发吞吐更高,归因新版 llama.cpp) | [json](../results/strix/qwen3-32b-bf16-llamacpp.json) |
| qwen3-32b-q4-k-m-llamacpp | llama.cpp | 10.87/91.93ms | 71.34 | ✅ 复现成功(单流 TPOT 锰点稳定;高并发吞吐更高,归因新版 llama.cpp) | [json](../results/strix/qwen3-32b-q4-k-m-llamacpp.json) |
| qwen3-32b-q8-0-llamacpp | llama.cpp | 6.46/154.9ms | 54.05 | ✅ 复现成功(单流 TPOT 锰点稳定;高并发吞吐更高,归因新版 llama.cpp) | [json](../results/strix/qwen3-32b-q8-0-llamacpp.json) |
| qwen3-4b-bf16-llamacpp | llama.cpp | 25.44/39.27ms | 262.2 | ✅ 复现成功(单流 TPOT 锰点稳定;高并发吞吐更高,归因新版 llama.cpp) | [json](../results/strix/qwen3-4b-bf16-llamacpp.json) |
| qwen3-4b-q4-k-m-llamacpp | llama.cpp | 67.48/14.8ms | 377.97 | ✅ 复现成功(单流 TPOT 锰点稳定;高并发吞吐更高,归因新版 llama.cpp) | [json](../results/strix/qwen3-4b-q4-k-m-llamacpp.json) |
| qwen3-4b-q8-0-llamacpp | llama.cpp | 45.13/22.12ms | 311.47 | ✅ 复现成功(单流 TPOT 锰点稳定;高并发吞吐更高,归因新版 llama.cpp) | [json](../results/strix/qwen3-4b-q8-0-llamacpp.json) |
| qwen3-8b-bf16-llamacpp | llama.cpp | 11.95/83.68ms | 125.19 | ✅ 复现成功(单流 TPOT 锰点稳定;高并发吞吐更高,归因新版 llama.cpp) | [json](../results/strix/qwen3-8b-bf16-llamacpp.json) |
| qwen3-8b-q4-k-m-llamacpp | llama.cpp | 40.33/24.76ms | 242.99 | ✅ 复现成功(单流 TPOT 锰点稳定;高并发吞吐更高,归因新版 llama.cpp) | [json](../results/strix/qwen3-8b-q4-k-m-llamacpp.json) |
| qwen3-8b-q8-0-llamacpp | llama.cpp | 25.82/38.7ms | 209.06 | ✅ 复现成功(单流 TPOT 锰点稳定;高并发吞吐更高,归因新版 llama.cpp) | [json](../results/strix/qwen3-8b-q8-0-llamacpp.json) |
| qwen3.6-27b-bf16-llamacpp | llama.cpp | 4.19/238.34ms | 31.51 | ✅ C=1 单流复现(命中带宽屋顶);⚠️ C=32 聚合低于 recipe 标称(如实标注) | [json](../results/strix/qwen3.6-27b-bf16-llamacpp.json) |
| qwen3.6-27b-q4-k-m-llamacpp | llama.cpp | 12.09/82.39ms | 47.38 | ✅ 复现成功(单流 TPOT 锰点稳定;高并发吞吐更高,归因新版 llama.cpp) | [json](../results/strix/qwen3.6-27b-q4-k-m-llamacpp.json) |
| qwen3.6-27b-q8-0-llamacpp | llama.cpp | 7.68/130.17ms | 41.77 | ✅ 复现成功(单流 TPOT 锰点稳定;高并发吞吐更高,归因新版 llama.cpp) | [json](../results/strix/qwen3.6-27b-q8-0-llamacpp.json) |
| qwen3.6-35b-a3b-bf16-llamacpp | llama.cpp | 23.63/41.86ms | 73.11 | ✅ 复现成功(单流 TPOT 锰点稳定;高并发吞吐更高,归因新版 llama.cpp) | [json](../results/strix/qwen3.6-35b-a3b-bf16-llamacpp.json) |
| qwen3.6-35b-a3b-q8-0-llamacpp | llama.cpp | 44.74/22.17ms | 130.99 | ✅ 复现成功(单流 TPOT 锰点稳定;高并发吞吐更高,归因新版 llama.cpp) | [json](../results/strix/qwen3.6-35b-a3b-q8-0-llamacpp.json) |
| qwen3.6-35b-a3b-ud-q4-k-m-llamacpp | llama.cpp | 49.83/19.93ms | 138.57 | ✅ 复现成功(单流 TPOT 锰点稳定;高并发吞吐更高,归因新版 llama.cpp) | [json](../results/strix/qwen3.6-35b-a3b-ud-q4-k-m-llamacpp.json) |
| step-3.5-flash-q4-k-s-llamacpp | llama.cpp | 23.55/42.41ms | — | ✅ 复现成功(需特殊内存配置,见下) | [json](../results/strix/step-3.5-flash-q4-k-s-llamacpp.json) |

## Cross-cutting findings

- **llama.cpp Q4/Q8** — C=1 TPOT is stable and scheduler-independent; high-concurrency
  throughput scales with the newer `fe7c8b2414` build (batches better). Version
  difference explained, not a regression.
- **vLLM bf16** — reproduced; 27B is a clean 1:1 per-concurrency match (same vLLM
  0.19.x), MoE bf16 (30B-A3B / gemma-26B) anchor matches and exceeds at C=32.
- **AWQ-4bit** — the recipe's quantized weights are not public; reproduced with
  public third-party weights (`cyankiwi/*`), so these are *runnable, not same-weight*.
- **Quark-W8A8-INT8** — weights are public (`nameistoken/*`, same path as the recipe).
  MoE (35B-A3B) and gemma-31B exceed the headline at C=32; **27B dense shows no INT8
  decode speedup** on this public image (Triton INT8 kernel ≈ BF16) — honestly below
  the recipe headline.
- **DiffusionGemma (bf16 + AWQ-INT4)** — needs the `vllm-main` image
  (`ghcr.io/radeon-arena/vllm-main:gfx1151`, upstream vLLM main) for the
  `DiffusionGemmaForBlockDiffusion` arch; the gfx11-line image cannot load it.
  Block-diffusion has very high TTFT (16–175 s) and emits no per-token stream
  (`tpot_ms` = null), as the recipe notes.
- **BF16-27B llama.cpp** — single-stream hits the memory-bandwidth roof (TPOT 238 ms
  ≈ vLLM bf16 227 ms) but C=32 (31.5) is below the recipe headline; BF16 batched GEMM
  in this llama.cpp build is weaker than its quantized paths.

