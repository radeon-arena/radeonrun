# Benchmark Results

Each file here is one benchmark run produced by:

```bash
python run-recipe.py <recipe> --benchmark benchmarking/<profile>.yaml --out results/<name>.json
```

A result is the reproducible output of `recipe × profile × gpu`. These JSON files
are the **source of truth** for the Radeon Arena leaderboard — the site ingests
this directory, not any private/internal pipeline.

## Result schema

```json
{
  "profile": "halo-arena-v1",
  "framework": "llama-benchy",
  "generated_at": "2026-06-19T08:00:00Z",
  "meta": {
    "recipe": "qwen3-6-35b-a3b-bf16",
    "model": "/models/Qwen3.6-35B-A3B",
    "runtime": "vllm",
    "container": "halo-vllm",
    "command": "vllm serve ... --attention-backend TRITON_ATTN",
    "gpu": "Radeon 8060S (Strix Halo, gfx1151)"
  },
  "measurements": [
    {
      "depth": 0, "concurrency": 1, "pp": 512, "tg": 128,
      "decode_toks_per_s": 124.3,
      "ttft_ms": 240.0,
      "tpot_ms": 8.0,
      "requests_ok": 1, "requests_total": 1
    }
  ]
}
```

| Field | Meaning |
|-------|---------|
| `profile` | Which benchmark profile produced these numbers. |
| `meta.recipe` / `meta.command` | The exact recipe + serve command (reproducibility). |
| `meta.gpu` | The Radeon GPU the run was measured on. |
| `measurements[]` | One row per `(depth, concurrency, pp, tg)` point. |
| `decode_toks_per_s` | Aggregate decode throughput. |
| `ttft_ms` / `tpot_ms` | Time-to-first-token / time-per-output-token (median). |

## Directory layout

```
results/
  strix/        # Radeon 8060S (Strix Halo, gfx1151)
  w7900/        # Radeon PRO W7900 (RDNA3, gfx1100)
  r9700/        # Radeon AI PRO R9700 (RDNA4, gfx1200)
  index.json    # generated manifest of all results (built by scripts/build-index.py)
```

Group result files under the hardware directory they were measured on. The site
maps these directories to its hardware tabs.
