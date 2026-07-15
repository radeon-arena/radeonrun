# Composable run configuration

RadeonRun models a benchmark as five independent axes:

```text
Run = Device × Model × Launch × Benchmark × Matrix
```

The configuration directories are:

| Axis | Location | Owns |
|---|---|---|
| Device | `devices/<id>.yaml` | GPU identity, architecture, runner labels and available runner capacity |
| Model | `models/catalog.yaml` or `models/<id>.yaml` | Artifact path, Hugging Face source/revision, quantization, served name and model metadata patches |
| Launch | `launches/catalog.yaml` or `launches/<id>.yaml` | Runtime, OCI image policy, defaults, environment, mods and serve command template |
| Benchmark | `benchmarking/<id>.yaml` | Prompt/decode sizes, context depths, concurrency, warmups, repetitions and schedule |
| Matrix | `matrices/catalog.yaml` or `matrices/<id>.yaml` | References that select and override the other four axes |

`run-recipe.py <id>` resolves a matrix first. If no matrix exists it loads the
legacy `recipes/<id>.yaml` and normalizes it to the same internal contract.
This keeps old links and third-party integrations working while new
configurations avoid copying model, launch and workload fields into one file.

## Matrix example

```yaml
matrices:
  qwen36-bf16-w7900-v2:
    model: qwen3.6-35b-a3b-bf16-vllm
    launch:
      ref: vllm-bf16-triton-no-gmu
      defaults:
        ctx: 131072
    device: w7900
    benchmark: halo-arena-v2
```

A matrix can override any model or launch field without changing the reusable
base spec:

```yaml
launch:
  ref: vllm-bf16-triton
  overrides:
    image:
      ref: quay.io/acme/optimized-vllm:v2
    env:
      MY_KERNEL_TOGGLE: "1"
```

Both `overrides:` and direct sibling fields are accepted; direct fields are a
compact shorthand.

## OCI images are registry-neutral

Images are not required to live in Radeon Arena GitHub Packages. A launch may
use any complete OCI reference:

```yaml
image:
  ref: docker.io/vllm/vllm-openai-rocm:latest
```

```yaml
image:
  ref: quay.io/acme/vllm@sha256:0123456789abcdef...
```

```yaml
image:
  by_device:
    halo: {tag: f5fa386fe}
    w7900: {ref: harbor.example.com/rocm/vllm:gfx1100}
  build:
    framework: vllm
```

Resolution precedence is:

1. CLI `--image <full-ref>`
2. launch `image.ref`
3. launch `image.repository` + tag
4. logical runtime fallback under `--registry`, `$RADEONRUN_IMAGE_REGISTRY`, or
   finally `ghcr.io/radeon-arena`

Explicit external images are pulled or used locally. A failed pull never causes
RadeonRun to build its own Dockerfile under the external image name. A launch
must explicitly declare `image.build` (or use the logical-runtime fallback) to
allow local building.

Pull behavior is controlled by:

```text
--pull-policy missing   # default: use local, else pull, else declared build
--pull-policy always    # refresh first; keep a present local image if refresh fails
--pull-policy never     # local image only
```

The runner records:

- `image_requested`: reference selected by configuration/CLI
- `image_resolved`: immutable repository digest returned by Docker, when present
- `image_digest`: digest portion of the resolved reference
- `image_id`: local Docker content ID
- `image_commit`: runtime source commit embedded by the image, when present

## Command templates

Launch commands may reference model fields and runtime defaults:

```yaml
command: >-
  vllm serve {model.path}
  --served-model-name {model.served_name}
  --host {host} --port {port}
  --max-num-seqs {nseq} --max-model-len {ctx}
```

Dotted spec placeholders such as `{model.path}` are resolved when the matrix is
loaded. Runtime placeholders such as `{port}`, `{nseq}` and `{ctx}` are resolved
later after CLI overrides and benchmark context limits have been applied.
The final rendered command—not only the template—is saved in new result files.

## Device and result keys

The CLI device IDs and result directories intentionally differ for Halo:

| Device ID | Result key | Architecture |
|---|---|---|
| `halo` | `strix` | `gfx1151` |
| `w7900` | `w7900` | `gfx1100` |
| `r9700` | `r9700` | `gfx1201` (`gfx1200` retained only as a compatibility alias) |

`device.topology.runner_capacity` describes how many GPUs are allocated to the
runner, not how many a result used. Actual tensor parallelism, visible-device
selection and multi-GPU flags belong to the Launch axis and are captured by the
rendered command/environment.

## Validation

Run all schema, compatibility, result and bundle checks with:

```bash
python3 scripts/build-results-bundle.py
python3 scripts/validate-configs.py
```

CI additionally runs:

```bash
python3 scripts/build-results-bundle.py --check
```

The check ignores only volatile generator metadata (`generated_at`, `commit`,
`short_commit`) and requires device lists, records, axes and provenance to match.

## Bundle v2

Every browser record in `results/bundle.json` contains:

```text
record
├── device
├── model
├── launch
├── benchmark
├── spec_files
├── data.measurements
├── recipe       # normalized compatibility view
└── data         # native result compatibility view
```

The website consumes these structured axes directly. It no longer needs to
infer a model source, image registry, device architecture or benchmark matrix
from a monolithic recipe string.
