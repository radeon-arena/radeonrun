#!/usr/bin/env python3
"""Configuration loader for RadeonRun's composable benchmark matrix.

New configurations split a run into independent axes:

* ``devices/<id>.yaml``    — hardware and runner metadata
* ``models/<id>.yaml``     — model artifact, source, revision and quantization
* ``launches/<id>.yaml``   — runtime, OCI image policy and serve command
* ``benchmarking/*.yaml``  — workload parameters
* ``matrices/<id>.yaml``   — references that compose the four axes

Legacy ``recipes/*.yaml`` remain supported.  They are normalized into the same
shape so every downstream consumer uses one contract while migration can happen
incrementally.
"""
from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any, Mapping

import yaml

ROOT = Path(__file__).resolve().parent
RECIPES_DIR = ROOT / "recipes"
MATRICES_DIR = ROOT / "matrices"
MODELS_DIR = ROOT / "models"
LAUNCHES_DIR = ROOT / "launches"
DEVICES_DIR = ROOT / "devices"
BENCHMARKING_DIR = ROOT / "benchmarking"

DEFAULT_IMAGE_REGISTRY = "ghcr.io/radeon-arena"
ENGINE_IMAGE = {"vllm": "vllm", "vllm-main": "vllm-main", "llamacpp": "llamacpp"}
_ENGINE_ALIASES = {
    "halo-vllm-opt": "vllm",
    "halo-vllm-main": "vllm-main",
    "halo-llamacpp": "llamacpp",
    "vllm-opt": "vllm",
    "vllm": "vllm",
    "vllm-main": "vllm-main",
    "llamacpp": "llamacpp",
}


class ConfigError(ValueError):
    """Raised when a composable run configuration is incomplete or invalid."""


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"configuration file not found: {path}")
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"configuration must be a mapping: {path}")
    return data


def deep_merge(base: Mapping[str, Any] | None, override: Mapping[str, Any] | None) -> dict[str, Any]:
    """Recursively merge mappings; scalar/list values in override replace base."""
    out: dict[str, Any] = copy.deepcopy(dict(base or {}))
    for key, value in dict(override or {}).items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _ref_and_overrides(value: Any, axis: str) -> tuple[str, dict[str, Any]]:
    if isinstance(value, str) and value.strip():
        return value.strip(), {}
    if not isinstance(value, Mapping):
        raise ConfigError(f"matrix {axis} must be a ref string or mapping")
    ref = str(value.get("ref") or "").strip()
    if not ref:
        raise ConfigError(f"matrix {axis}.ref is required")
    direct = {k: v for k, v in value.items() if k not in {"ref", "overrides"}}
    overrides = deep_merge(direct, value.get("overrides") if isinstance(value.get("overrides"), Mapping) else {})
    return ref, overrides


def _spec(path: Path, expected_id: str) -> dict[str, Any]:
    data = read_yaml(path)
    spec_id = str(data.get("id") or expected_id).strip()
    if spec_id != expected_id:
        raise ConfigError(f"{path}: id {spec_id!r} does not match filename {expected_id!r}")
    data["id"] = spec_id
    return data


def _catalog_spec(directory: Path, collection: str, spec_id: str) -> tuple[dict[str, Any], str]:
    """Load an individual spec file, falling back to ``catalog.yaml``."""
    path = directory / f"{spec_id}.yaml"
    if path.is_file():
        return _spec(path, spec_id), str(path.relative_to(ROOT))

    catalog_path = directory / "catalog.yaml"
    catalog = read_yaml(catalog_path)
    entries = catalog.get(collection)
    if not isinstance(entries, Mapping) or spec_id not in entries:
        raise ConfigError(f"configuration {spec_id!r} not found in {directory.relative_to(ROOT)}/")
    entry = entries[spec_id]
    if not isinstance(entry, Mapping):
        raise ConfigError(f"{catalog_path}: {collection}.{spec_id} must be a mapping")
    data = copy.deepcopy(dict(entry))
    data["id"] = spec_id
    return data, f"{catalog_path.relative_to(ROOT)}#{collection}.{spec_id}"


def _matrix(name: str) -> tuple[dict[str, Any], str] | tuple[None, None]:
    path = MATRICES_DIR / f"{name}.yaml"
    if path.is_file():
        data = read_yaml(path)
        matrix_id = str(data.get("id") or name).strip()
        if matrix_id != name:
            raise ConfigError(f"{path}: id {matrix_id!r} does not match filename {name!r}")
        data["id"] = name
        return data, str(path.relative_to(ROOT))

    catalog_path = MATRICES_DIR / "catalog.yaml"
    if not catalog_path.is_file():
        return None, None
    catalog = read_yaml(catalog_path)
    entries = catalog.get("matrices")
    if not isinstance(entries, Mapping) or name not in entries:
        return None, None
    entry = entries[name]
    if not isinstance(entry, Mapping):
        raise ConfigError(f"{catalog_path}: matrices.{name} must be a mapping")
    data = copy.deepcopy(dict(entry))
    data["id"] = name
    return data, f"{catalog_path.relative_to(ROOT)}#matrices.{name}"


def load_device_spec(device_id: str) -> tuple[dict[str, Any], Path]:
    path = DEVICES_DIR / f"{device_id}.yaml"
    device = _spec(path, device_id)
    for key in ("label", "gpu", "arch", "image_device"):
        if not device.get(key):
            raise ConfigError(f"{path}: missing {key}")
    return device, path


def load_benchmark_spec(profile: str | None) -> tuple[dict[str, Any], Path] | tuple[None, None]:
    if not profile:
        return None, None
    profile_name = Path(str(profile)).name
    if profile_name.endswith(".yaml"):
        profile_name = profile_name[:-5]
    path = BENCHMARKING_DIR / f"{profile_name}.yaml"
    data = read_yaml(path)
    data.setdefault("id", profile_name)
    data["file"] = f"benchmarking/{path.name}"
    return data, path


def _runtime_of(config: Mapping[str, Any]) -> str:
    raw = str(config.get("runtime") or config.get("container") or "").strip().lower()
    base = raw.split("/")[-1].split(":")[0]
    if raw in ENGINE_IMAGE:
        return raw
    if base in _ENGINE_ALIASES:
        return _ENGINE_ALIASES[base]
    if "vllm-main" in base:
        return "vllm-main"
    if "vllm" in base:
        return "vllm"
    if "llamacpp" in base or "llama-cpp" in base:
        return "llamacpp"
    return raw or "vllm"


def _lookup(context: Mapping[str, Any], dotted: str) -> Any:
    value: Any = context
    for part in dotted.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def render_known_placeholders(template: str, context: Mapping[str, Any]) -> str:
    """Render dotted spec placeholders while preserving runtime placeholders."""
    def replace(match: re.Match[str]) -> str:
        value = _lookup(context, match.group(1))
        return match.group(0) if value is None else str(value)

    return re.sub(r"\{([A-Za-z0-9_.-]+)\}", replace, template)


def render_command(config: Mapping[str, Any], overrides: Mapping[str, Any] | None = None) -> str:
    """Render runtime placeholders from launch defaults plus CLI overrides."""
    params = copy.deepcopy(dict(config.get("defaults") or {}))
    for key, value in dict(overrides or {}).items():
        if value is not None:
            params[key] = value
    command = str(config.get("command") or "").strip()
    return render_known_placeholders(command, params)


def _legacy_axes(recipe: dict[str, Any], device: dict[str, Any], benchmark: dict[str, Any] | None) -> tuple[dict, dict]:
    metadata = recipe.get("metadata") if isinstance(recipe.get("metadata"), Mapping) else {}
    model = {
        "id": str(recipe.get("_recipe_name") or recipe.get("name") or "model"),
        "name": recipe.get("name"),
        "path": recipe.get("model"),
        "source": recipe.get("source"),
        "revision": recipe.get("model_revision"),
        "served_name": recipe.get("served_model_name"),
        "quantization": metadata.get("quantization"),
        "description": recipe.get("description") or metadata.get("description"),
        "patches": recipe.get("model_patches") or [],
    }
    launch = {
        "id": str(recipe.get("_recipe_name") or recipe.get("name") or "launch"),
        "runtime": _runtime_of(recipe),
        "container": recipe.get("container"),
        "image": copy.deepcopy(recipe.get("image")),
        "image_tag": recipe.get("image_tag"),
        "defaults": copy.deepcopy(recipe.get("defaults") or {}),
        "env": copy.deepcopy(recipe.get("env") or {}),
        "mods": copy.deepcopy(recipe.get("mods") or []),
        "command": recipe.get("command"),
    }
    return model, launch


def _public_recipe(config: Mapping[str, Any]) -> dict[str, Any]:
    return {k: copy.deepcopy(v) for k, v in config.items() if not str(k).startswith("_")}


def load_run_config(
    name: str,
    *,
    device_override: str | None = None,
    benchmark_override: str | None = None,
) -> dict[str, Any]:
    """Load a matrix or legacy recipe and return one normalized recipe mapping.

    A matrix shadows a legacy recipe with the same name.  Internal ``_model``,
    ``_launch``, ``_device`` and ``_benchmark`` keys carry resolved axes; legacy
    flat keys are retained for existing runner functions and external tools.
    """
    legacy_path = RECIPES_DIR / f"{name}.yaml"
    matrix, matrix_file = _matrix(name)

    if matrix is not None:
        legacy = read_yaml(legacy_path) if legacy_path.is_file() else {}
        model_ref, model_overrides = _ref_and_overrides(matrix.get("model"), "model")
        launch_ref, launch_overrides = _ref_and_overrides(matrix.get("launch"), "launch")
        model_base, model_file = _catalog_spec(MODELS_DIR, "models", model_ref)
        launch_base, launch_file = _catalog_spec(LAUNCHES_DIR, "launches", launch_ref)
        model = deep_merge(model_base, model_overrides)
        launch = deep_merge(launch_base, launch_overrides)

        matrix_device = str(matrix.get("device") or "halo")
        device_id = str(device_override or matrix_device)
        device, device_path = load_device_spec(device_id)
        benchmark_ref = str(benchmark_override or matrix.get("benchmark") or "") or None
        benchmark, benchmark_path = load_benchmark_spec(benchmark_ref)

        context = {"model": model, "launch": launch, "device": device, "benchmark": benchmark or {}}
        command = render_known_placeholders(str(launch.get("command") or ""), context)
        legacy_metadata = legacy.get("metadata") if isinstance(legacy.get("metadata"), Mapping) else {}
        matrix_metadata = matrix.get("metadata") if isinstance(matrix.get("metadata"), Mapping) else {}
        metadata = deep_merge(legacy_metadata, matrix_metadata)
        if model.get("quantization") and not metadata.get("quantization"):
            metadata["quantization"] = model["quantization"]
        display_name = matrix.get("name") or legacy.get("name") or model.get("name") or name
        description = matrix.get("description") or legacy.get("description") or model.get("description") or display_name
        model.setdefault("name", display_name)
        model.setdefault("description", description)
        image = copy.deepcopy(launch.get("image"))

        config: dict[str, Any] = {
            "recipe_version": str(matrix.get("matrix_version") or "3"),
            "name": display_name,
            "description": description,
            "metadata": metadata,
            "model": model.get("path"),
            "source": model.get("source"),
            "model_revision": model.get("revision"),
            "served_model_name": model.get("served_name"),
            "model_patches": copy.deepcopy(model.get("patches") or []),
            "runtime": _runtime_of(launch),
            "container": launch.get("container") or _runtime_of(launch),
            "image": image,
            "image_tag": launch.get("image_tag"),
            "benchmark_ctx": matrix.get("benchmark_ctx", model.get("benchmark_ctx", legacy.get("benchmark_ctx"))),
            "mods": copy.deepcopy(launch.get("mods") or []),
            "defaults": copy.deepcopy(launch.get("defaults") or {}),
            "env": copy.deepcopy(launch.get("env") or {}),
            "command": command,
            "_recipe_name": name,
            "_config_source": "matrix",
            "_matrix": matrix,
            "_model": model,
            "_launch": launch,
            "_device": device,
            "_benchmark": benchmark,
            "_spec_files": {
                "matrix": matrix_file,
                "model": model_file,
                "launch": launch_file,
                "device": str(device_path.relative_to(ROOT)),
                "benchmark": str(benchmark_path.relative_to(ROOT)) if benchmark_path else None,
                "legacy_recipe": str(legacy_path.relative_to(ROOT)) if legacy_path.is_file() else None,
            },
        }
        return config

    if not legacy_path.is_file():
        raise ConfigError(f"run configuration not found: {name} (checked matrices/ and recipes/)")

    recipe = read_yaml(legacy_path)
    recipe["_recipe_name"] = name
    device_id = str(device_override or recipe.get("device") or "halo")
    device, device_path = load_device_spec(device_id)
    benchmark_ref = benchmark_override or recipe.get("benchmark")
    benchmark, benchmark_path = load_benchmark_spec(str(benchmark_ref) if benchmark_ref else None)
    model, launch = _legacy_axes(recipe, device, benchmark)
    recipe["_config_source"] = "legacy"
    recipe["_model"] = model
    recipe["_launch"] = launch
    recipe["_device"] = device
    recipe["_benchmark"] = benchmark
    recipe["_spec_files"] = {
        "matrix": None,
        "model": None,
        "launch": None,
        "device": str(device_path.relative_to(ROOT)),
        "benchmark": str(benchmark_path.relative_to(ROOT)) if benchmark_path else None,
        "legacy_recipe": str(legacy_path.relative_to(ROOT)),
    }
    return recipe


def list_run_configs() -> list[str]:
    names = {p.stem for p in RECIPES_DIR.glob("*.yaml")}
    names.update(p.stem for p in MATRICES_DIR.glob("*.yaml"))
    catalog_path = MATRICES_DIR / "catalog.yaml"
    if catalog_path.is_file():
        entries = read_yaml(catalog_path).get("matrices")
        if isinstance(entries, Mapping):
            names.update(str(name) for name in entries)
    names.discard("catalog")
    return sorted(names)


def _image_config(recipe: Mapping[str, Any], device: str) -> tuple[dict[str, Any], bool]:
    raw = recipe.get("image")
    if raw is None and isinstance(recipe.get("_launch"), Mapping):
        raw = recipe["_launch"].get("image")
    if isinstance(raw, str):
        return {"ref": raw}, True
    if not isinstance(raw, Mapping):
        return {}, False

    config = copy.deepcopy(dict(raw))
    by_device = config.pop("by_device", {})
    selected = by_device.get(device) if isinstance(by_device, Mapping) else None
    if isinstance(selected, str):
        config = deep_merge(config, {"ref": selected})
    elif isinstance(selected, Mapping):
        config = deep_merge(config, selected)
    explicit = bool(config.get("ref") or config.get("repository"))
    return config, explicit


def _replace_tag(image: str, tag: str | None) -> str:
    if not tag or "@" in image:
        return image
    slash = image.rfind("/")
    colon = image.rfind(":")
    base = image[:colon] if colon > slash else image
    return f"{base}:{tag}"


def image_tag(image: str | None) -> str | None:
    if not image:
        return None
    if "@" in image:
        return image.split("@", 1)[1]
    slash = image.rfind("/")
    colon = image.rfind(":")
    return image[colon + 1:] if colon > slash else "latest"


def resolve_image(
    recipe: Mapping[str, Any],
    device: str,
    *,
    tag_override: str | None = None,
    image_override: str | None = None,
    registry: str | None = None,
) -> str:
    """Resolve the requested OCI image without forcing Radeon Arena packages.

    Precedence: ``--image`` > explicit launch ``image.ref``/``repository`` >
    default image derived from runtime and device.  An explicit OCI reference is
    preserved verbatim unless the caller deliberately supplies ``--tag``.
    """
    loaded_device = recipe.get("_device") if isinstance(recipe.get("_device"), Mapping) else {}
    if loaded_device.get("id") == device:
        device_spec = loaded_device
    else:
        device_spec, _ = load_device_spec(device)
    context = {
        "device": device_spec,
        "model": recipe.get("_model") if isinstance(recipe.get("_model"), Mapping) else {},
        "launch": recipe.get("_launch") if isinstance(recipe.get("_launch"), Mapping) else {},
    }
    if image_override:
        return render_known_placeholders(str(image_override), context)

    image_cfg, explicit = _image_config(recipe, device)
    ref = image_cfg.get("ref")
    if ref:
        return _replace_tag(render_known_placeholders(str(ref), context), tag_override)

    repository = image_cfg.get("repository")
    if repository:
        repository = render_known_placeholders(str(repository), context).rstrip(":")
        tag = tag_override or image_cfg.get("tag") or "latest"
        return f"{repository}:{tag}"

    raw_container = str(recipe.get("container") or "").strip()
    if "/" in raw_container:
        return _replace_tag(raw_container, tag_override)

    runtime = _runtime_of(recipe)
    image_name = ENGINE_IMAGE.get(runtime)
    if not image_name:
        if raw_container:
            return _replace_tag(raw_container, tag_override)
        raise ConfigError(f"cannot resolve image for runtime {runtime!r}")

    image_device = str(device_spec.get("image_device") or device)
    registry = (registry or os.getenv("RADEONRUN_IMAGE_REGISTRY") or DEFAULT_IMAGE_REGISTRY).rstrip("/")
    legacy_pin = recipe.get("image_tag") if recipe.get("_config_source") != "matrix" and device == "halo" else None
    tag = str(tag_override or image_cfg.get("tag") or legacy_pin or "latest")
    return f"{registry}/{image_device}-{image_name}:{tag}"


def image_is_explicit(recipe: Mapping[str, Any], device: str) -> bool:
    image_cfg, explicit = _image_config(recipe, device)
    return explicit or "/" in str(recipe.get("container") or "")


def image_build_config(recipe: Mapping[str, Any], device: str) -> dict[str, Any] | None:
    image_cfg, explicit = _image_config(recipe, device)
    build = image_cfg.get("build")
    if isinstance(build, Mapping):
        if build.get("enabled", True) is False:
            return None
        return copy.deepcopy(dict(build))
    if build is True:
        return {"framework": _runtime_of(recipe)}
    if explicit:
        return None
    return {"framework": _runtime_of(recipe)}


def public_recipe(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the normalized compatibility recipe without private loader keys."""
    return _public_recipe(config)


def resolved_axes(config: Mapping[str, Any], image: str | None = None) -> dict[str, Any]:
    """Return JSON-safe device/model/launch/benchmark axes for results/bundles."""
    device = copy.deepcopy(config.get("_device") or {})
    model = copy.deepcopy(config.get("_model") or {})
    launch = copy.deepcopy(config.get("_launch") or {})
    benchmark = copy.deepcopy(config.get("_benchmark") or {})
    launch["runtime"] = launch.get("runtime") or _runtime_of(config)
    launch["container"] = launch.get("container") or config.get("container")
    launch["image"] = image or launch.get("image")
    launch["image_tag"] = image_tag(str(launch.get("image") or "")) or launch.get("image_tag")
    launch["defaults"] = copy.deepcopy(config.get("defaults") or launch.get("defaults") or {})
    launch["env"] = copy.deepcopy(config.get("env") or launch.get("env") or {})
    launch["mods"] = copy.deepcopy(config.get("mods") or launch.get("mods") or [])
    launch["command"] = config.get("command") or launch.get("command")
    launch["config_source"] = config.get("_config_source", "legacy")
    launch["spec_files"] = copy.deepcopy(config.get("_spec_files") or {})
    return {"device": device, "model": model, "launch": launch, "benchmark": benchmark}
