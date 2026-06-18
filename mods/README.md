# Mods and Patches

A **mod** is a directory containing a `run.sh` (and optional patch/template
files) that is applied at container build or launch time to fix a specific
model or tweak the runtime.

> **SCAFFOLD:** the two mods below are skeletons. Real per-model fixes for
> ROCm/Strix Halo go here as they are discovered and verified on hardware.

## Convention

```
mods/<mod-name>/
  run.sh            # executed with $WORKSPACE_DIR set; applies the fix
  *.patch / *.jinja # optional supporting files
```

A recipe references mods by path:

```yaml
mods:
  - mods/fix-qwen3.6-chat-template
```

## Current mods (scaffold)

- `use-official-vllm/`        — compatibility shim for official vLLM images (TODO)
- `fix-qwen3.6-chat-template/` — install a corrected chat template (skeleton)
