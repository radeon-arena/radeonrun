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


def _make_prompt(approx_tokens: int, salt: str = "") -> str:
    """Deterministic ~approx_tokens-token English prompt (~9 tokens/sentence).

    Matches the reproduce harness (bench_stream.py) so prefill/decode behavior —
    and therefore the numbers — are comparable to the published leaderboard. A
    degenerate prompt like "token token ..." instead skews tokenization and can
    trigger early EOS, so it is deliberately avoided.
    """
    sentence = "The quick brown fox jumps over the lazy dog. "
    reps = max(1, approx_tokens // 9 + 1)
    text = (sentence * reps).strip()
    return f"{text} {salt}".strip() if salt else text


def _make_measured_prompt(depth: int, pp: int, salt: str = "") -> str:
    """Build a prompt with a reusable prefix plus a measured suffix.

    When `depth > 0`, the prefix is stable across warm-up and timed requests so
    engines with prefix caching can reuse it. The suffix varies by request/run to
    avoid turning the measured `pp` segment itself into a full-prompt cache hit.
    """
    prefix = _make_prompt(depth, "shared prefix") if depth > 0 else ""
    suffix = _make_prompt(pp, salt)
    return f"{prefix}\n{suffix}" if prefix else suffix


def _post_stream_text(base_url: str, model: str, prompt: str, max_tokens: int):
    """Issue one streaming completion; return (ttft_s, decode_s, out_tokens).

    ttft_s    : time to first streamed token.
    decode_s  : first->last token wall time (pure decode window; excludes TTFT
                and the trailing [DONE]/socket-close latency).
    out_tokens: the server's reported completion_tokens when available, else the
                streamed-chunk count.
    These definitions match reproduce/bench_stream.py so results are comparable.
    """
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.0,
        "ignore_eos": True,                          # force exactly max_tokens of decode
        "stream_options": {"include_usage": True},   # get exact completion_tokens
    }).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    t_first = None
    t_last = t0
    n_chunks = 0
    usage = None
    with urllib.request.urlopen(req, timeout=900) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except ValueError:
                continue
            choices = chunk.get("choices") or []
            if choices and choices[0].get("text"):
                now = time.perf_counter()
                if t_first is None:
                    t_first = now
                t_last = now
                n_chunks += 1
            if chunk.get("usage"):
                usage = chunk["usage"]
    out_tokens = (usage or {}).get("completion_tokens") or n_chunks
    if t_first is None:
        return None, 0.0, 0
    return t_first - t0, t_last - t_first, out_tokens


def _post_stream(base_url: str, model: str, depth: int, pp: int, tg: int, salt: str = ""):
    prompt = _make_measured_prompt(depth, pp, salt)
    return _post_stream_text(base_url, model, prompt, tg)


def _prime_prefix(base_url: str, model: str, depth: int) -> None:
    if depth <= 0:
        return
    _post_stream_text(base_url, model, _make_prompt(depth, "shared prefix"), 1)


def _run_concurrent(base_url: str, model: str, depth: int, pp: int, tg: int, concurrency: int, prefix_caching: bool, run_id: int):
    """Run `concurrency` streaming requests in parallel; aggregate metrics."""
    results: list[tuple[float, float, int]] = []
    lock = threading.Lock()

    if prefix_caching and depth > 0:
        try:
            _prime_prefix(base_url, model, depth)
        except Exception as exc:  # noqa: BLE001
            print(f"    prefix prime failed: {exc}", file=sys.stderr)

    def worker(worker_id: int):
        try:
            r = _post_stream(base_url, model, depth, pp, tg, salt=f"run {run_id} worker {worker_id}")
            with lock:
                results.append(r)
        except Exception as exc:  # noqa: BLE001 — record failure, keep going
            with lock:
                results.append((float("nan"), float("nan"), 0))
            print(f"    request failed: {exc}", file=sys.stderr)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(concurrency)]
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
    # Aggregate decode throughput = all generated tokens / wall time (= reproduce
    # agg_decode_tps). Per-request decode and TPOT use only the first->last decode
    # window (exclude prefill/TTFT), matching reproduce/bench_stream.py. Aggregate
    # across the concurrent requests with the mean, as reproduce does.
    decode_toks_s = total_out / wall if wall > 0 else 0.0
    per_dec, tpots, prefills = [], [], []
    for _ttft, decode_s, n in ok:
        if n > 1 and decode_s > 0:
            step_s = decode_s / (n - 1)
            per_dec.append((n - 1) / decode_s)
            tpots.append(step_s * 1000.0)
            prefill_s = _ttft - step_s
            if prefill_s > 0:
                prefills.append(pp / prefill_s)
    return {
        "decode_toks_per_s": round(decode_toks_s, 2),
        "decode_toks_per_s_per_req": round(statistics.mean(per_dec), 2) if per_dec else None,
        "prefill_toks_per_s": round(statistics.mean(prefills), 2) if prefills else None,
        "ttft_ms": round(statistics.mean(ttfts) * 1000.0, 2),
        "tpot_ms": round(statistics.mean(tpots), 2) if tpots else None,
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
    prefix_caching = bool(args.get("prefix_caching", False))

    # Honor an explicit heat-aware schedule if present; otherwise sweep the grid.
    schedule = profile.get("schedule")
    if schedule:
        points = [(int(s["depth"]), int(s["concurrency"])) for s in schedule]
    else:
        points = [(d, c) for d in depths for c in concs]

    measurements = []
    failed_points = 0
    for depth, conc in points:
        for pp in pps:
            for tg in tgs:
                # Warm-up (discarded).
                for _ in range(warmup):
                    try:
                        if prefix_caching:
                            _prime_prefix(base_url, model, depth)
                        _post_stream(base_url, model, depth, pp, tg, salt="warmup")
                    except Exception:  # noqa: BLE001
                        pass
                # Timed runs; keep the median point.
                run_results = []
                for run_idx in range(runs):
                    r = _run_concurrent(base_url, model, depth, pp, tg, conc, prefix_caching, run_idx)
                    if r:
                        run_results.append(r)
                if not run_results:
                    print(f"  depth={depth} c={conc} pp={pp} tg={tg}: all runs failed", file=sys.stderr, flush=True)
                    failed_points += 1
                    continue
                # Median by decode throughput.
                run_results.sort(key=lambda x: x["decode_toks_per_s"])
                med = run_results[len(run_results) // 2]
                med.update({"depth": depth, "concurrency": conc, "pp": pp, "tg": tg})
                measurements.append(med)
                print(f"  depth={depth:6d} c={conc:2d} pp={pp} tg={tg}: "
                      f"decode={med['decode_toks_per_s']:.1f} tok/s "
                        f"ttft={med['ttft_ms']:.0f}ms tpot={med['tpot_ms']}", flush=True)

    return {
        "profile": (profile.get("metadata") or {}).get("name", "unknown"),
        "framework": profile.get("framework", "halo-arena"),
        "measurements": measurements,
        "failed_points": failed_points,
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
          f"with profile {profile.get('metadata', {}).get('name', args.profile)}", flush=True)

    result = run_profile(args.base_url, args.model, profile)
    if result.get("failed_points"):
        print(f"Benchmark failed: {result['failed_points']} profile points produced no valid runs", file=sys.stderr, flush=True)
        return 1
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
        print(f"Wrote {args.out} ({len(result['measurements'])} points)", flush=True)
    else:
        print(out_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
