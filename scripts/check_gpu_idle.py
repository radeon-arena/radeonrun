#!/usr/bin/env python3
"""Fail when a Radeon benchmark device is already under load."""
from __future__ import annotations

import argparse
import re
import subprocess
import sys


def gpu_use_values(text: str) -> list[int]:
    return [int(value) for value in re.findall(r"GPU use \(%\):\s*(\d+)", text)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-use", type=int, default=10)
    parser.add_argument("--expected-gpus", type=int, default=1)
    args = parser.parse_args()

    probe = subprocess.run(
        ["rocm-smi", "--showuse"], capture_output=True, text=True, timeout=60,
        check=False,
    )
    if probe.returncode != 0:
        print(probe.stderr.strip() or "rocm-smi --showuse failed", file=sys.stderr)
        return 2
    values = gpu_use_values(probe.stdout)
    if len(values) < args.expected_gpus:
        print(
            f"expected at least {args.expected_gpus} GPUs, found {len(values)} in rocm-smi output",
            file=sys.stderr,
        )
        return 2
    busy = [(index, value) for index, value in enumerate(values[:args.expected_gpus]) if value > args.max_use]
    if busy:
        print(
            "GPU workload already active: "
            + ", ".join(f"GPU{index}={value}%" for index, value in busy),
            file=sys.stderr,
        )
        return 1
    print("GPU idle check ok: " + ", ".join(f"GPU{index}={value}%" for index, value in enumerate(values[:args.expected_gpus])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())