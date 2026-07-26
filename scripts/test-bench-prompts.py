#!/usr/bin/env python3
"""Focused regression tests for token-bounded benchmark prompts."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("bench", ROOT / "bench.py")
assert SPEC and SPEC.loader
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


def main() -> int:
    bench._PROMPT_CACHE.clear()
    with mock.patch.object(bench, "_token_count", side_effect=lambda _url, _model, prompt: len(prompt)):
        first = bench._make_prompt_bounded("http://test", "model", 96, "c16 run0 worker0")
        second = bench._make_prompt_bounded("http://test", "model", 96, "c16 run0 worker1")
        repeated = bench._make_prompt_bounded("http://test", "model", 96, "c16 run0 worker0")

    assert len(first) == 96
    assert len(second) == 96
    assert first != second
    assert first[:40] != second[:40]
    assert first == repeated
    assert first.startswith("Benchmark request ")

    events: list[str] = []

    def make_prompt(*_args, **_kwargs):
        events.append("prompt")
        return "measured prompt"

    def post_stream(*_args, **_kwargs):
        events.append("request")
        return 0.1, 1.27, 128, 512

    def perf_counter():
        events.append("timer")
        return 0.0 if events.count("timer") == 1 else 1.0

    with mock.patch.object(bench, "_make_measured_prompt", side_effect=make_prompt), \
            mock.patch.object(bench, "_token_count", return_value=512), \
            mock.patch.object(bench, "_post_stream_text", side_effect=post_stream), \
            mock.patch.object(bench.time, "perf_counter", side_effect=perf_counter):
        measurement = bench._run_concurrent(
            "http://test", "model", 0, 512, 128, 1, False, 0, 768,
        )
    assert events.index("prompt") < events.index("timer") < events.index("request")
    assert measurement is not None
    assert measurement["prompt_tokens"] == 512
    assert measurement["prompt_tokens_min"] == 512
    assert measurement["prompt_tokens_max"] == 512
    assert measurement["prefill_toks_per_s"] == round(512 / 0.09, 2)
    with mock.patch.object(bench, "_run_concurrent", return_value=measurement):
        result = bench.run_profile(
            "http://test",
            "model",
            {
                "metadata": {"name": "test"},
                "args": {
                    "depth": [0],
                    "pp": [512],
                    "tg": [128],
                    "concurrency": [1],
                    "warmup": 0,
                    "runs": 1,
                },
            },
            max_context=768,
        )
    assert result["methodology"] == bench.BENCHMARK_METHODOLOGY
    print("token-bounded benchmark prompts are unique and deterministic")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())