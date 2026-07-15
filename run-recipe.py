#!/usr/bin/env python3
"""
run-recipe.py - One-click composable matrix runner.

Resolves Device + Model + Launch + Benchmark specs (or a legacy flat recipe),
fills the command template with defaults plus CLI overrides, and runs it via
launch-cluster.sh --solo (or prints it with --print).

Examples:
  ./run-recipe.py --list
  ./run-recipe.py qwen3.6-35b-a3b-bf16 --print
  MODELS_DIR=/models ./run-recipe.py qwen3.6-35b-a3b-bf16
"""

import argparse
import functools
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
from pathlib import Path

from radeonrun_config import (
    ConfigError,
    image_build_config,
    image_is_explicit,
    image_tag,
    list_run_configs,
    load_run_config,
    render_command,
    resolve_image,
    resolved_axes,
)

RECIPES_DIR = Path(__file__).resolve().parent / "recipes"
MODEL_CONTEXTS = Path(__file__).resolve().parent / "benchmarking" / "model-contexts.yaml"

# Image registry + device profiles. A recipe names a logical engine
# (vllm | vllm-main | llamacpp); the concrete image is
#     ghcr.io/radeon-arena/<device>-<engine-image>:<tag>
# so the device is in the name and the tag carries the build version (a commit
# id for byte-reproducible pins, or the `latest` moving tag for convenience).
# Device id -> GPU arch. The device id is also the image-name prefix
# (halo = Strix Halo / Radeon 8060S / gfx1151).
DEVICE_GFX = {"halo": "gfx1151", "w7900": "gfx1100", "r9700": "gfx1201"}
# Container names seen in recipes (logical or legacy) -> logical engine.
_ENGINE_ALIASES = {
    "halo-vllm-opt": "vllm",
    "halo-vllm-main": "vllm-main",
    "halo-llamacpp": "llamacpp",
    "vllm-opt": "vllm",
    "vllm": "vllm",
    "vllm-main": "vllm-main",
    "llamacpp": "llamacpp",
}


@functools.lru_cache(maxsize=1)
def _docker_cmd() -> tuple[str, ...]:
    """Return a working Docker command for the current runner user."""
    explicit = os.environ.get("RADEONRUN_DOCKER", "").strip()
    candidates = [shlex.split(explicit)] if explicit else [["docker"], ["sudo", "-n", "docker"]]
    for candidate in candidates:
        if not candidate or shutil.which(candidate[0]) is None:
            continue
        try:
            probe = subprocess.run(
                [*candidate, "info"], stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, timeout=30, check=False,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if probe.returncode == 0:
            return tuple(candidate)
    hint = f" from RADEONRUN_DOCKER={explicit!r}" if explicit else ""
    raise RuntimeError(f"no usable Docker command{hint}; tried docker and sudo -n docker")


def _docker(*args: str) -> list[str]:
    return [*_docker_cmd(), *args]


def _engine_of(recipe, container):
    """Logical engine for a recipe: explicit `runtime`, else its container name."""
    rt = str(recipe.get("runtime") or "").strip().lower()
    if rt in ("vllm", "vllm-main", "llamacpp"):
        return rt
    base = container.split("/")[-1].split(":")[0]  # strip registry path + tag
    if base in _ENGINE_ALIASES:
        return _ENGINE_ALIASES[base]
    if base.startswith("halo-vllm") or base in ("vllm", "vllm-opt"):
        return "vllm"
    if "llamacpp" in base or "llama-cpp" in base:
        return "llamacpp"
    return None


def _resolve_container(
    recipe: dict,
    device: str,
    tag: str | None = None,
    image: str | None = None,
    registry: str | None = None,
) -> str:
    """Resolve a normalized launch to an OCI image reference.

    Explicit image references are first-class and are never rewritten to
    Radeon Arena packages.  The Radeon Arena registry is only the fallback for
    logical runtimes that do not declare an image.
    """
    return resolve_image(
        recipe,
        device,
        tag_override=tag,
        image_override=image,
        registry=registry,
    )


def _image_provenance(container: str) -> dict:
    """Record the concrete build identity of a serve image for the result meta.

    Radeon Arena is a performance leaderboard, so a number is only meaningful
    paired with the exact image that produced it. Capture the image ref, its
    local sha256 id, and the build commit baked in at /app/commit.txt (the
    llama.cpp / vLLM source commit) so a leaderboard entry pins to a real,
    reproducible build instead of a moving `:latest` tag. Best-effort: any
    piece that cannot be resolved is simply omitted.
    """
    prov = {"image": container, "image_requested": container}
    prov["image_tag"] = image_tag(container) or "latest"
    try:
        r = subprocess.run(
            _docker("image", "inspect", container),
            capture_output=True, text=True, timeout=30)
        if r.returncode == 0 and r.stdout.strip():
            inspected = json.loads(r.stdout)[0]
            if inspected.get("Id"):
                prov["image_id"] = inspected["Id"]
            digests = inspected.get("RepoDigests") or []
            if digests:
                requested_repo = container.split("@", 1)[0]
                requested_repo = requested_repo.rsplit(":", 1)[0] if ":" in requested_repo.rsplit("/", 1)[-1] else requested_repo
                resolved = next((d for d in digests if d.split("@", 1)[0] == requested_repo), digests[0])
                prov["image_resolved"] = resolved
                prov["image_digest"] = resolved.split("@", 1)[1] if "@" in resolved else resolved
    except Exception:  # noqa: BLE001
        pass
    for path in ("/app/commit.txt", "/commit.txt"):
        try:
            r = subprocess.run(
                _docker("run", "--rm", "--pull=never", "--entrypoint", "cat", container, path),
                capture_output=True, text=True, timeout=60)
            if r.returncode == 0 and r.stdout.strip():
                prov["image_commit"] = r.stdout.strip().splitlines()[0].strip()
                break
        except Exception:  # noqa: BLE001
            pass
    return prov


def _host_models_dir() -> Path:
    """Host models dir that launch-cluster.sh bind-mounts to /models."""
    return Path(os.environ.get("MODELS_DIR", "/models"))


def _host_model_path(model: str) -> Path:
    """Map a recipe's in-container model path (/models/...) onto the host dir."""
    rel = model[len("/models/"):] if model.startswith("/models/") else model.lstrip("/")
    return _host_models_dir() / rel


def _gguf_shard_glob(fname: str):
    """If `fname` is one shard of a split gguf, return the glob for all shards."""
    m = re.match(r"(.+)-\d{5}-of-\d{5}\.gguf$", fname)
    return (m.group(1) + "-*-of-*.gguf") if m else None


def _gguf_fetch_shape(host_path: Path) -> tuple[str, Path]:
    """Return (HF include pattern, local-dir) for a recipe GGUF path.

    Local model paths are laid out as `/models/<model-dir>/<repo-path>`. For a
    flat repo file that means `<repo-path>` is just the basename; for split BF16
    files it can be `BF16/Foo-00001-of-00002.gguf`; for MiMo it is
    `UD-Q2_K_XL/Foo-00001-of-00004.gguf`. The HF include pattern must be the
    repo path, while `--local-dir` is the local model directory.
    """
    rel = host_path.relative_to(_host_models_dir())
    parts = rel.parts
    if len(parts) >= 3:
        model_dir = _host_models_dir() / parts[0]
        repo_path = "/".join(parts[1:])
    else:
        model_dir = host_path.parent
        repo_path = host_path.name
    return (_gguf_shard_glob(repo_path) or repo_path, model_dir)


def _directory_model_complete(host_path: Path) -> bool:
    """True when a staged HF directory has enough files to serve.

    A failed/partial HF download can leave a directory with only config.json or
    one shard. Treating that as staged causes vLLM to fail later with missing
    tokenizer or checkpoint-shard errors, so validate the minimal HF layout here.
    """
    if not host_path.is_dir():
        return False
    if not (host_path / "config.json").is_file():
        return False
    has_tokenizer = any(
        (host_path / name).is_file()
        for name in ("tokenizer.json", "tokenizer.model", "vocab.json", "merges.txt")
    )
    if not has_tokenizer:
        return False

    index_path = host_path / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            data = json.loads(index_path.read_text())
            shards = set(data.get("weight_map", {}).values())
        except Exception:  # noqa: BLE001
            return False
        return bool(shards) and all((host_path / shard).is_file() for shard in shards)

    return any(host_path.glob("*.safetensors")) or any(host_path.glob("*.bin"))


def _fetch_plan(recipe: dict):
    """Download plan for a recipe: (repo, include, dest_dir, revision) or None.

    Derived from the recipe's `source` (HF repo id) + `model` (in-container
    path). The download shape is inferred from the model path:
      - a *.gguf file  -> fetch that file (or every shard of a split gguf)
      - a directory    -> fetch the whole repo into it
    Returns None when the recipe declares no `source` (legacy / pre-staged).
    """
    source = str(recipe.get("source") or "").strip()
    model = str(recipe.get("model") or "").strip()
    if not source or not model:
        return None
    revision = str(recipe.get("model_revision") or "").strip() or None
    host_path = _host_model_path(model)
    if model.endswith(".gguf"):
        include, dest_dir = _gguf_fetch_shape(host_path)
        return (source, include, dest_dir, revision)
    return (source, None, host_path, revision)


def _model_present(model: str) -> bool:
    """True when the recipe's model already exists under host MODELS_DIR."""
    model = str(model or "").strip()
    if not model:
        return False
    host_path = _host_model_path(model)
    if model.endswith(".gguf"):
        if host_path.exists():
            return True
        glob = _gguf_shard_glob(host_path.name)
        return bool(glob and list(host_path.parent.glob(glob)))
    return _directory_model_complete(host_path)


def _ensure_model_available(model: str) -> bool:
    """Validate a staged model path and create a convenience symlink for shards."""
    host_path = _host_model_path(model)
    if not model.endswith(".gguf"):
        return _directory_model_complete(host_path)
    if host_path.exists():
        return True
    glob = _gguf_shard_glob(host_path.name)
    if not glob:
        return False
    shards = sorted(host_path.parent.glob(glob))
    if not shards:
        return False
    # llama.cpp can expand split GGUFs from the first shard, but recipe paths may
    # refer to an unsharded convenience name. Point that name at shard 1.
    try:
        host_path.symlink_to(shards[0].name)
    except FileExistsError:
        pass
    return host_path.exists()


def _hf_cli():
    import shutil
    for c in ("hf", "huggingface-cli"):
        if shutil.which(c):
            return c
    return None


def ensure_image(container: str, recipe: dict, device: str,
                 build: bool = True, pull: bool = True, push: bool = False,
                 pull_policy: str = "missing") -> int:
    """Ensure the serve image exists locally; pull or build it if missing.

    Resolution order: local image -> `docker pull` (when `pull`) -> build from
    dockerfiles/ via build.sh (when `build`). build.sh names the image exactly
    like `_resolve_container`, so a local source build satisfies the run with no
    registry at all -- this is what makes the *image* self-contained too. When
    `push`, a freshly built image is synced back to ghcr so other runners can
    just pull it.

    A commit-pinned tag (anything but the `latest` moving tag) is treated as
    pull-only: building it from the current dockerfiles/ source would yield a
    different binary mislabeled with that commit and silently corrupt the
    leaderboard pin, so `--build` is ignored for pinned tags.
    """
    null = subprocess.DEVNULL
    build_config = image_build_config(recipe, device)
    if build_config is None:
        build = False
        # Explicit third-party images have no local Dockerfile contract.
        pull = True
    if pull_policy == "always":
        pull = True
    present = subprocess.call(_docker("image", "inspect", container),
                              stdout=null, stderr=null) == 0
    if pull_policy == "always" and pull:
        print(f"[image] refreshing {container}")
        if subprocess.call(_docker("pull", container)) == 0:
            return 0
        if present:
            print(f"[image] refresh failed; keeping local image: {container}", file=sys.stderr)
            return 0
    if present:
        print(f"[image] present: {container}")
        return 0
    if pull_policy == "never":
        print(f"[image] missing locally and pull_policy=never: {container}", file=sys.stderr)
        return 1

    last = container.rsplit("/", 1)[-1]
    if ":" in last and last.rsplit(":", 1)[-1] != "latest":
        # Commit-pinned: the only faithful source is the registry build.
        if build and not pull:
            print(f"[image] {container} is commit-pinned; forcing pull, not a source build")
        pull, build = True, False
    if pull:
        print(f"[image] pulling {container}")
        if subprocess.call(_docker("pull", container)) == 0:
            return 0
        print("[image] pull failed; building from dockerfiles/ instead")
    if not build:
        origin = "external image has no build recipe" if image_is_explicit(recipe, device) else "build disabled"
        print(f"[image] missing: {container} ({origin})", file=sys.stderr)
        return 1
    engine = str((build_config or {}).get("framework") or _engine_of(recipe, container) or "vllm")
    build_sh = RECIPES_DIR.parent / "build.sh"
    cmd = [str(build_sh), "-f", engine, "-d", device, "-t", container]
    if push:
        cmd.append("--push")  # sync the freshly built image back to ghcr
    print(f"[image] building: {' '.join(cmd[1:])}")
    return subprocess.call(cmd)


def setup_model(recipe: dict, force: bool = False) -> int:
    """Stage a recipe's model from its HF `source` into the host MODELS_DIR.

    Idempotent: skips when the model is already present unless `force`. This is
    the `setup` half of the self-contained runner pipeline.
    """
    plan = _fetch_plan(recipe)
    if plan is None:
        print("[setup] recipe declares no `source`; assuming model is pre-staged")
        return 0
    repo, include, dest_dir, revision = plan
    model = str(recipe.get("model") or "")
    if not force and _model_present(model):
        print(f"[setup] already staged: {model}")
        return 0
    cli = _hf_cli()
    if cli is None:
        print("[setup] hf CLI not found. Install: pip install --user 'huggingface_hub[cli]'",
              file=sys.stderr)
        return 1
    import subprocess
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    cmd = [cli, "download", repo, "--local-dir", str(dest_dir)]
    if include:
        cmd += ["--include", include]
    if revision:
        cmd += ["--revision", revision]
    env = dict(os.environ)
    print(f"[setup] {' '.join(cmd)}")
    rc = subprocess.call(cmd, env=env)
    if rc != 0:
        return rc
    if not _ensure_model_available(model):
        print(f"[setup] downloaded zero usable files for {model} (include={include})", file=sys.stderr)
        return 1
    return 0


def teardown_model(recipe: dict) -> int:
    """Delete the model staged for this recipe, freeing host disk.

    Only touches paths under the host MODELS_DIR derived from the recipe's
    `model`. This is the `teardown` half of the pipeline.
    """
    model = str(recipe.get("model") or "").strip()
    if not model:
        return 0
    import shutil
    host_path = _host_model_path(model)
    if model.endswith(".gguf"):
        glob = _gguf_shard_glob(host_path.name)
        targets = sorted(host_path.parent.glob(glob)) if glob else [host_path]
        for t in targets:
            try:
                t.unlink()
            except FileNotFoundError:
                pass
            except OSError as e:
                print(f"[teardown] failed to remove {t}: {e}", file=sys.stderr)
        # hf download leaves a .cache/ metadata dir under --local-dir; clear it
        cache = host_path.parent / ".cache"
        if cache.is_dir():
            shutil.rmtree(cache, ignore_errors=True)
        try:
            host_path.parent.rmdir()  # drop the containing dir if now empty
        except OSError:
            pass
        print(f"[teardown] removed staged gguf for {model}")
    else:
        shutil.rmtree(host_path, ignore_errors=True)
        print(f"[teardown] removed {host_path}")
    return 0


def list_recipes() -> None:
    found = list_run_configs()
    if not found:
        print("No recipes found.")
        return
    print("Available recipes:")
    for name in found:
        print(f"  - {name}")


def _render_command(recipe: dict, overrides: dict) -> str:
    """Fill the recipe's command template with defaults + CLI overrides."""
    return render_command(recipe, overrides)


def _find_free_port(preferred: int = 8000) -> int:
    """Return a free localhost port, preferring the recipe/default port."""
    def available(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", port))
            except OSError:
                return False
        return True

    if available(preferred):
        return preferred
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _profile_required_ctx(profile_path: str | None) -> int | None:
    """Return the minimum server context length required by a benchmark profile."""
    if not profile_path:
        return None
    import yaml

    profile = yaml.safe_load(Path(profile_path).read_text()) or {}
    args = profile.get("args") or {}

    def values(name: str, default: int) -> list[int]:
        raw = args.get(name, [default])
        if not isinstance(raw, list):
            raw = [raw]
        return [int(v) for v in raw]

    depths = values("depth", 0)
    schedule = profile.get("schedule") or []
    if isinstance(schedule, list):
        depths.extend(int(p["depth"]) for p in schedule if isinstance(p, dict) and "depth" in p)

    return max(depths or [0]) + max(values("pp", 512)) + max(values("tg", 128))


def _benchmark_context_metadata() -> dict:
    if not MODEL_CONTEXTS.is_file():
        return {}
    import yaml
    return yaml.safe_load(MODEL_CONTEXTS.read_text()) or {}


def _recipe_name_from_path(recipe: dict) -> str | None:
    name = str(recipe.get("name") or "").strip()
    # Historical recipe names omit the runtime suffix; workflow recipes use the
    # file stem. Prefer an explicit helper field if present, else caller fills it.
    return str(recipe.get("_recipe_name") or name or "").strip() or None


def _model_ctx_for_recipe(recipe: dict) -> int | None:
    meta = _benchmark_context_metadata()
    recipe_name = _recipe_name_from_path(recipe)
    if recipe_name:
        raw = (meta.get("recipes") or {}).get(recipe_name, {}).get("model_ctx")
        if raw is not None:
            return int(raw)
    source = str(recipe.get("source") or "").strip()
    raw = (meta.get("sources") or {}).get(source, {}).get("model_ctx")
    return int(raw) if raw is not None else None


def _recipe_benchmark_ctx(recipe: dict) -> int | None:
    """Per-request context limit to enforce while measuring a profile."""
    meta = _benchmark_context_metadata()
    recipe_name = _recipe_name_from_path(recipe)
    if recipe_name:
        raw = (meta.get("recipes") or {}).get(recipe_name, {}).get("benchmark_ctx")
        if raw is not None:
            return int(raw)
    raw = recipe.get("benchmark_ctx")
    if raw is not None:
        return int(raw)
    return _model_ctx_for_recipe(recipe)


def _serve_ctx_for_recipe(recipe: dict, container: str, nseq: int | None = None,
                          effective_ctx: int | None = None) -> int | None:
    """Context value rendered into the serve command.

    vLLM `--max-model-len` is per request. llama.cpp server `-c` is total KV
    cache shared across `-np` slots, so to support a per-request benchmark cap at
    concurrency N it needs roughly cap*N total context. Clamp to the model
    context if no benchmark cap is known.
    """
    benchmark_ctx = effective_ctx or _recipe_benchmark_ctx(recipe)
    defaults = recipe.get("defaults") or {}
    default_ctx = int(defaults["ctx"]) if defaults.get("ctx") is not None else None
    if _engine_of(recipe, container) != "llamacpp":
        if benchmark_ctx is None:
            return default_ctx
        return max(default_ctx or 0, benchmark_ctx)
    per_request = benchmark_ctx or default_ctx
    if per_request is None:
        return None
    slots = int(nseq or defaults.get("nseq") or 1)
    return max(default_ctx or 0, per_request * max(1, slots))


def _env_prefix(recipe: dict) -> str:
    """Shell prefix that exports recipe env vars before the serve command."""
    import shlex
    env = recipe.get("env") or {}
    if not isinstance(env, dict):
        return ""
    parts = []
    for k, v in env.items():
        if not k or v is None:
            continue
        parts.append(f"{k}={shlex.quote(str(v))}")
    return " ".join(parts)


def _apply_model_patches(recipe: dict) -> None:
    """Apply small, explicit recipe-declared patches to staged model metadata."""
    patches = recipe.get("model_patches") or []
    if not isinstance(patches, list):
        return
    model = str(recipe.get("model") or "").strip()
    if not model:
        return
    host_path = _host_model_path(model)
    for patch in patches:
        if not isinstance(patch, dict):
            continue
        if patch.get("type") not in {"set_quant_method", "set_quant_config", "set_config"}:
            continue
        cfg = host_path / "config.json"
        if not cfg.exists():
            continue
        import json
        original = cfg.read_text()
        data = json.loads(original)
        changes = {}
        if patch.get("type") == "set_quant_method":
            qcfg = data.setdefault("quantization_config", {})
            changes["quant_method"] = patch.get("value")
            target = qcfg
        elif patch.get("type") == "set_quant_config":
            qcfg = data.setdefault("quantization_config", {})
            values = patch.get("values") or {}
            if isinstance(values, dict):
                changes.update(values)
            target = qcfg
        else:
            values = patch.get("values") or {}
            if isinstance(values, dict):
                changes.update(values)
            target = data
        changes = {k: v for k, v in changes.items() if k and v is not None}
        if changes and any(target.get(k) != v for k, v in changes.items()):
            backup = cfg.with_suffix(".json.orig")
            if not backup.exists():
                backup.write_text(original)
            before = {k: target.get(k) for k in changes}
            target.update(changes)
            cfg.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
            print(f"[setup] patched {cfg}: {before!r} -> {changes!r}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="One-click composable benchmark runner for AMD Radeon GPUs.",
    )
    parser.add_argument("recipe", nargs="?", help="Recipe name (without .yaml)")
    parser.add_argument("--list", action="store_true", help="List available recipes")
    parser.add_argument("--solo", action="store_true", help="Single-node mode (default)")
    parser.add_argument("--print", dest="print_only", action="store_true",
                        help="Print the launch command instead of running it")
    parser.add_argument("--port", type=int, help="Override serve port")
    parser.add_argument("--nseq", type=int, help="Override --max-num-seqs / -np")
    parser.add_argument("--ctx", type=int, help="Override llama.cpp -c context")
    parser.add_argument("--benchmark", metavar="PROFILE",
                        help="Benchmark profile YAML to run against the served endpoint "
                             "(e.g. benchmarking/halo-arena-v1.yaml)")
    parser.add_argument("--out", help="Output JSON path for --benchmark results")
    parser.add_argument("--base-url", default="http://localhost:8000",
                        help="Endpoint to benchmark (default: http://localhost:8000)")
    parser.add_argument("--device", default=None, choices=sorted(DEVICE_GFX),
                        help="Target GPU device profile (default: matrix device, else halo)")
    parser.add_argument("--tag", default=None,
                        help="Override the selected image tag for a one-off run")
    parser.add_argument("--image", default=None,
                        help="Explicit OCI image reference. Used verbatim and never rewritten "
                             "to the Radeon Arena registry.")
    parser.add_argument("--registry", default=None,
                        help="Default registry for logical runtimes without an explicit image "
                             "(default: $RADEONRUN_IMAGE_REGISTRY or ghcr.io/radeon-arena)")
    parser.add_argument("--pull-policy", choices=("always", "missing", "never"), default="missing",
                        help="Container image pull policy (default: missing)")
    parser.add_argument("--setup-only", action="store_true",
                        help="Only stage the model from its HF source, then exit")
    parser.add_argument("--no-setup", action="store_true",
                        help="Skip model staging (assume the model is already present)")
    parser.add_argument("--force-setup", action="store_true",
                        help="Re-download the model even if it is already staged")
    parser.add_argument("--cleanup", action="store_true",
                        help="Delete the staged model after the run (frees host disk)")
    parser.add_argument("--hf-token",
                        help="HuggingFace token for gated/private model repos (else $HF_TOKEN)")
    parser.add_argument("--build", action="store_true",
                        help="Build the serve image from dockerfiles/ instead of pulling it")
    parser.add_argument("--no-build", action="store_true",
                        help="Never build the image; only use a local or pulled one")
    parser.add_argument("--push", action="store_true",
                        help="After building an image, push it to ghcr (needs docker login ghcr.io)")
    args = parser.parse_args()

    if args.list or not args.recipe:
        list_recipes()
        return 0

    try:
        recipe = load_run_config(
            args.recipe,
            device_override=args.device,
            benchmark_override=args.benchmark,
        )
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    args.device = str(args.device or (recipe.get("_device") or {}).get("id") or "halo")
    if args.image:
        # Make the CLI override part of the effective launch spec so image
        # preparation knows it is external and never falls back to our
        # Dockerfiles when a pull fails.
        recipe["image"] = {"ref": args.image}
        if isinstance(recipe.get("_launch"), dict):
            recipe["_launch"]["image"] = {"ref": args.image}
    default_port = int(args.port or (recipe.get("defaults") or {}).get("port", 8000))
    run_port = _find_free_port(default_port) if args.benchmark else default_port
    if args.benchmark and run_port != default_port:
        print(f"[benchmark] port {default_port} is busy; using {run_port}")
    profile_ctx = _profile_required_ctx(args.benchmark) if args.benchmark else None
    container = _resolve_container(recipe, args.device, args.tag, args.image, args.registry)
    benchmark_ctx = _recipe_benchmark_ctx(recipe) if args.benchmark else None
    if benchmark_ctx is not None:
        profile_ctx = min(profile_ctx, benchmark_ctx) if profile_ctx else benchmark_ctx
    run_ctx = args.ctx if args.ctx is not None else _serve_ctx_for_recipe(recipe, container, args.nseq, profile_ctx)
    if args.benchmark and profile_ctx:
        if benchmark_ctx is not None:
            print(f"[benchmark] effective context length = {profile_ctx} (recipe benchmark_ctx={benchmark_ctx})")
        else:
            print(f"[benchmark] profile requires context length >= {profile_ctx}")
    cmd = _render_command(recipe, {"port": run_port, "nseq": args.nseq, "ctx": run_ctx})
    if not cmd:
        print(f"Recipe '{args.recipe}' has no command.")
        return 2

    # Build the launch-cluster.sh invocation. The recipe command already has
    # the model path baked in; we just wrap it in the solo launcher.
    launch = (RECIPES_DIR.parent / "launch-cluster.sh")
    port = run_port
    inner = cmd.replace("\\\n", " ")        # drop line-continuation backslashes
    inner = " ".join(inner.split())          # collapse whitespace
    prefix = _env_prefix(recipe)
    if prefix:
        inner = f"env {prefix} {inner}"
    if _engine_of(recipe, container) == "llamacpp" and "LD_LIBRARY_PATH" not in (recipe.get("env") or {}):
        inner = f"env LD_LIBRARY_PATH=/app {inner}"
    # Own the container name so teardown removes the exact container we launched
    # (and distinct recipes don't collide on a shared default name).
    container_name = os.environ.get("CONTAINER") or f"radeonrun_{args.recipe}"
    full = f'CONTAINER={container_name} IMAGE={container} {launch} --solo -p {port}:{port} exec {inner}'

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
    img_build, img_pull = not args.no_build, not args.build

    # --setup-only: stage everything (image + model) and stop.
    if args.setup_only:
        rc = ensure_image(container, recipe, args.device, build=img_build, pull=img_pull,
                  push=args.push, pull_policy=args.pull_policy)
        if rc != 0:
            return rc
        return setup_model(recipe, force=args.force_setup)

    # Print mode shouldn't touch the network or disk, even when showing the
    # benchmark-specific rendered serve command.
    if args.print_only:
        print(full)
        return 0

    # Self-contained pipeline: prepare the image (local -> pull -> build from
    # dockerfiles/) and stage the model from its `source` before serving, so a
    # recipe reproduces from nothing but this repo + a HuggingFace pull.
    if not args.no_setup:
        rc = ensure_image(container, recipe, args.device, build=img_build, pull=img_pull,
                  push=args.push, pull_policy=args.pull_policy)
        if rc != 0:
            print("[setup] image prepare failed; aborting", file=sys.stderr)
            return rc
        rc = setup_model(recipe, force=args.force_setup)
        if rc != 0:
            print("[setup] model staging failed; aborting", file=sys.stderr)
            return rc
        _apply_model_patches(recipe)

    # Benchmark mode: serve in the background, run the profile, then report.
    if args.benchmark:
        rc = _benchmark(recipe, full, args, container, container_name, port=run_port,
                max_context=benchmark_ctx, rendered_command=inner)
        if args.cleanup:
            teardown_model(recipe)
        return rc

    print(full)
    if args.print_only:
        return 0

    import subprocess
    _free_page_cache(args.device)
    rc = subprocess.call(full, shell=True)
    if args.cleanup:
        teardown_model(recipe)
    return rc


def _served_model_name(recipe: dict) -> str:
    """Best-effort served model name for the benchmark client."""
    declared = str(recipe.get("served_model_name") or "").strip()
    if declared:
        return declared
    cmd = recipe.get("command") or ""
    import re
    m = re.search(r"--served-model-name(?:=|\s+)(\S+)", cmd) or re.search(r"--alias(?:=|\s+)(\S+)", cmd)
    if m:
        return m.group(1).rstrip("\\")
    model = str(recipe.get("model", ""))
    return model or "model"


def _free_page_cache(device: str = "halo") -> None:
    """Drop the OS page cache before serving.

    On Strix Halo (gfx1151) the GPU is an APU: its memory is carved from system
    RAM (GTT). The model files we just staged sit in the page cache, which counts
    against the "free" memory vLLM probes at startup, so a faithful
    `--gpu-memory-utilization` can spuriously fail the memory check. Dropping the
    reclaimable cache restores the GTT headroom. Best-effort: needs passwordless
    sudo, harmless no-op on discrete-GPU devices.
    """
    if device not in ("halo",):
        return
    import subprocess
    try:
        subprocess.run(["sync"], check=False)
        subprocess.run(["sudo", "-n", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"],
                       check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["free", "-h"], check=False)
    except Exception:  # noqa: BLE001
        pass


def _benchmark(recipe: dict, serve_cmd: str, args, container: str,
               container_name: str = "radeon_vllm", port: int | None = None,
               max_context: int | None = None,
               rendered_command: str | None = None) -> int:
    """Serve the recipe in the background, run the profile, tear down."""
    import subprocess
    import time
    import urllib.request

    here = Path(__file__).resolve().parent
    model_name = _served_model_name(recipe)
    port = int(port or args.port or (recipe.get("defaults") or {}).get("port", 8000))
    base_url = args.base_url if args.base_url != "http://localhost:8000" else f"http://localhost:{port}"

    _free_page_cache(args.device)
    print(f"[benchmark] starting server: {recipe.get('name', args.recipe)}")
    server = subprocess.Popen(serve_cmd, shell=True)
    try:
        # Wait for /v1/models to come up (up to ~30 min for large BF16 GGUFs).
        ready = False
        for _ in range(1800):
            if server.poll() is not None:
                print("[benchmark] server exited before becoming ready")
                return 1
            try:
                with urllib.request.urlopen(base_url.rstrip('/') + "/v1/models", timeout=3):
                    ready = True
                    break
            except Exception:  # noqa: BLE001
                time.sleep(1)
        if not ready:
            print("[benchmark] server did not become ready")
            return 1
        print("[benchmark] server ready; running profile")

        out = args.out or str(here / "results" / f"{args.recipe}.json")
        # A directory `--out` (e.g. the workflow passes `results/`) means "write
        # <recipe>.json in here"; bench.py expects a concrete file path.
        if out.endswith(os.sep) or os.path.isdir(out):
            out = os.path.join(out, f"{args.recipe}.json")
        axes = resolved_axes(recipe, container)
        provenance = _image_provenance(container)
        axes["launch"].update(provenance)
        axes["launch"]["command"] = rendered_command or recipe.get("command")
        meta_obj = {
            "recipe": recipe.get("name", args.recipe),
            "model": recipe.get("model"),
            "runtime": recipe.get("runtime") or _engine_of(recipe, container) or "vllm",
            "container": recipe.get("container"),
            "command": " ".join((rendered_command or recipe.get("command") or "").split()),
            "config_source": recipe.get("_config_source", "legacy"),
            "spec_files": recipe.get("_spec_files") or {},
            "device": axes["device"],
            "model_spec": axes["model"],
            "launch_spec": axes["launch"],
            "benchmark_spec": axes["benchmark"],
        }
        # Pin the leaderboard number to the exact image build that produced it.
        meta_obj.update(provenance)
        meta = json.dumps(meta_obj)
        bench_cmd = [
            sys.executable, str(here / "bench.py"),
            "--base-url", base_url,
            "--model", model_name,
            "--profile", args.benchmark,
            "--out", out,
            "--meta", meta,
        ]
        if max_context is not None:
            bench_cmd.extend(["--max-context", str(max_context)])
        rc = subprocess.call(bench_cmd)
        if rc != 0:
            return rc
        try:
            data = json.loads(Path(out).read_text())
            if not data.get("measurements"):
                print("[benchmark] profile produced zero measurements; treating as failure", file=sys.stderr)
                return 1
        except Exception as exc:  # noqa: BLE001
            print(f"[benchmark] could not validate result file: {exc}", file=sys.stderr)
            return 1
        return 0
    finally:
        print("[benchmark] stopping server")
        server.terminate()
        try:
            server.wait(timeout=30)
        except Exception:  # noqa: BLE001
            server.kill()
        # `docker run --rm` only removes the container once it EXITS; terminating
        # the client above leaves the detached server container running (holding
        # the name + port), which would poison the NEXT recipe's benchmark. Remove
        # the exact container we launched (run-recipe owns the name via CONTAINER).
        subprocess.run(_docker("rm", "-f", container_name),
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)


if __name__ == "__main__":
    sys.exit(main())
