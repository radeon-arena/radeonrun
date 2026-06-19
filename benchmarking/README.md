# Benchmark Profiles

Standardized benchmarking profile definitions for **halo-arena** / `run-recipe.py`.
These profiles define the *exact* test grid every result on
[Radeon Arena](https://github.com/radeon-arena/radeon-docker) is measured
against, so any number on the leaderboard is reproducible by anyone.

A profile is the reproducibility contract: it fixes the input/output shapes,
context depths, concurrency levels, warm-up, and repeat count. Two people running
the same recipe against the same profile on the same GPU should get matching
numbers (within run-to-run noise).

## Profiles

| File | Framework | Depths | pp / tg | Concurrency | Runs | Use |
|------|-----------|--------|---------|-------------|------|-----|
| `halo-arena-v1.yaml` | llama-benchy | 0 | 512 / 128 | 1, 4, 16, 32 | 3 | Quick daily sweep (the in512/out128 streaming scenario) |
| `halo-arena-v2.yaml` | llama-benchy | 0 … 100000 | 2048 / 128 | 1, 2, 5, 10 | 3 | Full long-context sweep |

## How a profile is consumed

```bash
# Serve a recipe, then benchmark it against a profile:
python run-recipe.py recipes/qwen3.6-35b-a3b-bf16-vllm.yaml \
  --benchmark benchmarking/halo-arena-v1.yaml \
  --out results/
```

`run-recipe.py` reads the profile, drives the OpenAI-compatible `/v1/completions`
endpoint at each `(depth, pp, tg, concurrency)` point, repeats `runs` times after
a warm-up, and records streaming throughput, TTFT and TPOT from the SSE
timestamps. Output is one JSON document per `(recipe × profile × gpu)`.

## Why the schedule is shuffled (heat-aware)

Like Spark Arena's profiles, `halo-arena-v2.yaml` carries an explicit `schedule:`
that interleaves heavy and light points instead of running them in sorted order.
Running all high-concurrency / deep-context points back-to-back lets the GPU heat
soak and depresses later numbers; interleaving keeps thermal load even so results
are comparable. The schedule is generated with a fixed seed so the *ordering
itself* is reproducible.

## Field reference

| Field | Meaning |
|-------|---------|
| `framework` | Measurement harness (`llama-benchy`). |
| `args.depth` | Prefilled context depth(s) in tokens before the timed request (`0` = cold). |
| `args.pp` | Prefill (prompt) length in tokens for the prefill metric. |
| `args.tg` | Decode (generation) length in tokens for the decode metric. |
| `args.concurrency` | Parallel request counts to sweep. |
| `args.prefix_caching` | Whether prefix caching is enabled during the sweep. |
| `args.runs` | Repeats per measured point (median is reported). |
| `args.warmup` | Warm-up requests discarded before timing. |
| `schedule` | Optional explicit, heat-aware ordering of `(depth, concurrency)` points. |
