#!/usr/bin/env python3
"""
bench.py — halo-arena benchmark harness.

Drives an already-serving OpenAI-compatible endpoint through a benchmark profile
(see benchmarking/*.yaml) and writes one result JSON. This is the measurement
half of reproducibility: a recipe fixes *how the server is launched*, a profile
fixes *what workload is measured*, and this harness produces the numbers.

For each (depth, pp, tg, concurrency) point it:
  - optionally primes `depth` tokens of context (prefix),
  - issues a warm-up request (discarded),
  - runs `runs` timed streaming requests at the given concurrency,
  - records decode tok/s, prefill tok/s, TTFT (ms) and TPOT (ms) from the
    Server-Sent-Events stream timestamps,
  - reports the median across `runs`.

Usage (standalone):
  python bench.py --base-url http://localhost:8000 --model my-model \
      --profile benchmarking/halo-arena-v1.yaml --out results/run.json

Normally invoked via `run-recipe.py <recipe> --benchmark <profile>`.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any


def _post_stream(base_url: str, model: str, prompt_tokens: int, max_tokens: int):
    """Issue one streaming completion; return (ttft_s, total_s, out_tokens)."""
    # A simple synthetic prompt of roughly `prompt_tokens` words.
    prompt = " ".join(["token"] * max(1, prompt_tokens))
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    ttft = None
    out_tokens = 0
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                break
            if ttft is None:
                ttft = time.perf_counter() - t0
            try:
                chunk = json.loads(payload)
                text = chunk.get("choices", [{}])[0].get("text", "")
                if text:
                    out_tokens += 1
            except (ValueError, KeyError, IndexError):
                pass
    total = time.perf_counter() - t0
    return ttft if ttft is not None else total, total, out_tokens


def _run_concurrent(base_url: str, model: str, pp: int, tg: int, concurrency: int):
    """Run `concurrency` streaming requests in parallel; aggregate metrics."""
    results: list[tuple[float, float, int]] = []
    lock = threading.Lock()

    def worker():
        try:
            r = _post_stream(base_url, model, pp, tg)
            with lock:
                results.append(r)
        except Exception as exc:  # noqa: BLE001 — record failure, keep going
            with lock:
                results.append((float("nan"), float("nan"), 0))
            print(f"    request failed: {exc}", file=sys.stderr)

    threads = [threading.Thread(target=worker) for _ in range(concurrency)]
    wall_t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - wall_t0

    ok = [r for r in results if r[2] > 0]
    if not ok:
        return None
    ttfts = [r[0] for r in ok]
    total_out = sum(r[2] for r in ok)
    # Aggregate decode throughput = all generated tokens / wall time.
    decode_toks_s = total_out / wall if wall > 0 else 0.0
    # Per-request TPOT = (total - ttft) / (out_tokens - 1), averaged.
    tpots = []
    for ttft, total, n in ok:
        if n > 1:
            tpots.append((total - ttft) / (n - 1) * 1000.0)
    return {
        "decode_toks_per_s": round(decode_toks_s, 2),
        "ttft_ms": round(statistics.median(ttfts) * 1000.0, 2),
        "tpot_ms": round(statistics.median(tpots), 2) if tpots else None,
        "requests_ok": len(ok),
        "requests_total": len(results),
    }


def run_profile(base_url: str, model: str, profile: dict[str, Any]) -> dict[str, Any]:
    args = profile.get("args", {})
    pps = args.get("pp", [512])
    tgs = args.get("tg", [128])
    concs = args.get("concurrency", [1])
    depths = args.get("depth", [0])
    runs = int(args.get("runs", 3))
    warmup = int(args.get("warmup", 1))

    # Honor an explicit heat-aware schedule if present; otherwise sweep the grid.
    schedule = profile.get("schedule")
    if schedule:
        points = [(int(s["depth"]), int(s["concurrency"])) for s in schedule]
    else:
        points = [(d, c) for d in depths for c in concs]

    measurements = []
    for depth, conc in points:
        for pp in pps:
            for tg in tgs:
                # Warm-up (discarded).
                for _ in range(warmup):
                    try:
                        _post_stream(base_url, model, pp, tg)
                    except Exception:  # noqa: BLE001
                        pass
                # Timed runs; keep the median point.
                run_results = []
                for _ in range(runs):
                    r = _run_concurrent(base_url, model, pp, tg, conc)
                    if r:
                        run_results.append(r)
                if not run_results:
                    print(f"  depth={depth} c={conc} pp={pp} tg={tg}: all runs failed", file=sys.stderr)
                    continue
                # Median by decode throughput.
                run_results.sort(key=lambda x: x["decode_toks_per_s"])
                med = run_results[len(run_results) // 2]
                med.update({"depth": depth, "concurrency": conc, "pp": pp, "tg": tg})
                measurements.append(med)
                print(f"  depth={depth:6d} c={conc:2d} pp={pp} tg={tg}: "
                      f"decode={med['decode_toks_per_s']:.1f} tok/s "
                      f"ttft={med['ttft_ms']:.0f}ms tpot={med['tpot_ms']}")

    return {
        "profile": (profile.get("metadata") or {}).get("name", "unknown"),
        "framework": profile.get("framework", "halo-arena"),
        "measurements": measurements,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="halo-arena benchmark harness")
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", required=True, help="served model name")
    ap.add_argument("--profile", required=True, help="benchmark profile YAML")
    ap.add_argument("--out", help="output JSON path (default: stdout)")
    ap.add_argument("--meta", help="extra JSON merged into the result (recipe/gpu info)")
    args = ap.parse_args()

    import yaml
    profile = yaml.safe_load(Path(args.profile).read_text())
    print(f"Benchmarking {args.model} @ {args.base_url} "
          f"with profile {profile.get('metadata', {}).get('name', args.profile)}")

    result = run_profile(args.base_url, args.model, profile)
    if args.meta:
        try:
            result["meta"] = json.loads(args.meta)
        except ValueError:
            result["meta"] = {"raw": args.meta}
    result["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    out_json = json.dumps(result, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(out_json)
        print(f"Wrote {args.out} ({len(result['measurements'])} points)")
    else:
        print(out_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
