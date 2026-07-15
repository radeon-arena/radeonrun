#!/usr/bin/env python3
"""Focused regression tests for Docker command selection."""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SPEC = importlib.util.spec_from_file_location("run_recipe", ROOT / "run-recipe.py")
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def result(code: int) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess([], code)


def main() -> int:
    runner._docker_cmd.cache_clear()
    with mock.patch.dict(os.environ, {}, clear=False), \
            mock.patch.object(runner.shutil, "which", return_value="/usr/bin/cmd"), \
            mock.patch.object(runner.subprocess, "run", side_effect=[result(1), result(0)]):
        os.environ.pop("RADEONRUN_DOCKER", None)
        assert runner._docker_cmd() == ("sudo", "-n", "docker")
        assert runner._docker("pull", "image") == ["sudo", "-n", "docker", "pull", "image"]

    runner._docker_cmd.cache_clear()
    with mock.patch.dict(os.environ, {"RADEONRUN_DOCKER": "podman --remote"}), \
            mock.patch.object(runner.shutil, "which", return_value="/usr/bin/podman"), \
            mock.patch.object(runner.subprocess, "run", return_value=result(0)):
        assert runner._docker_cmd() == ("podman", "--remote")

    print("docker command selection ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())