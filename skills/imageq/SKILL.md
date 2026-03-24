---
name: imageq
description: "Load, smoke-test, or torchao-quantize a local diffusers image/editing pipeline from disk only (local_files_only). Use when the user mentions diffusers, image generation, image editing, Qwen-Image, DiT/transformer pipelines, loading a pipeline from disk, validating weights on GPU, submodule overrides (UNet/transformer/VAE), torchao quantization, W8A8, W4A16, or paths under MODELSCOPE_CACHE — and the model is already on disk. Do not use for downloads or ModelScope API calls (use msdl first). Do not use for causal LLM quantization (use llmq). No TensorRT. Runs in tmux."
metadata:
  {
    "openclaw":
      {
        "emoji": "🖼️",
        "requires": { "bins": ["uv", "tmux"] },
      },
  }
---

# imageq — local diffusers pipelines

Run `scripts/run.py` or `scripts/quantize.py` on the **target host** in tmux. Prerequisite: pipeline files already exist (typically after **msdl**). `quantize.py` produces a quantized component directory; `run.py` is the validation/acceptance step that mounts that component back into the base pipeline and records a test report.

## Agent behavior

Run **imageq** in a **detached tmux session** so the chat does not block.

Start `run.py`:

```bash
tmux new-session -d -s imageq-<short-name> \
  "VENV={baseDir}/.venv && (test -d \$VENV || uv venv \$VENV) && uv pip install -p \$VENV -r {baseDir}/requirements.txt && MODELSCOPE_CACHE=\${MODELSCOPE_CACHE:-/workspace/models} \$VENV/bin/python {baseDir}/scripts/run.py <model-path> [flags]; exec bash"
```

Start `quantize.py`:

```bash
tmux new-session -d -s imageq-<short-name> \
  "VENV={baseDir}/.venv && (test -d \$VENV || uv venv \$VENV) && uv pip install -p \$VENV -r {baseDir}/requirements.txt && MODELSCOPE_CACHE=\${MODELSCOPE_CACHE:-/workspace/models} \$VENV/bin/python {baseDir}/scripts/quantize.py <model-path> [flags]; exec bash"
```

- Session name: `imageq-` + short lowercase slug (e.g. `imageq-qwen-image-edit`).
- Do **not** chain **msdl** and **imageq** in one shell string; use two sessions when both are needed.
- After starting, tell the user the session name and stop. Poll logs only if they ask.

Progress:

```bash
tmux capture-pane -t imageq-<short-name> -p | tail -50
```

Success marker: `Done.` in the pane.

Full log:

```bash
tmux capture-pane -t imageq-<short-name> -p -S -
```

## Paths

- Relative `model-path` resolves under `MODELSCOPE_CACHE` (default `/workspace/models`), same convention as **llmq** and **msdl**.
- If the directory is missing, run **msdl** in a separate session first (`skills/msdl/SKILL.md`).

## Quick start

```
model-path: Qwen/Qwen-Image-Edit-2511
session:    imageq-qwen-image-edit
```

## Examples

Quantize the transformer with default `W8A8` (`torchao int8dq`):

```
session:    imageq-qwen-image-edit
model-path: Qwen/Qwen-Image-Edit-2511 --target transformer --scheme W8A8
```

Quantize the transformer with `W4A16` (`torchao int4wo`):

```
session:    imageq-qwen-image-edit
model-path: Qwen/Qwen-Image-Edit-2511 --target transformer --scheme W4A16 --group-size 128
```

Smoke (moves pipeline to CUDA when available):

```
session:    imageq-qwen-image-edit
model-path: Qwen/Qwen-Image-Edit-2511 --smoke
```

Validate a quantized transformer folder:

```
model-path: Qwen/Qwen-Image-Edit-2511 --transformer path/to/quantized-transformer --override-weight-format pytorch --smoke --report path/to/quantized-transformer/run-report.json
```

Overrides load the base pipeline first, then replace each requested submodule using that component’s concrete class (`type(pipe.unet)` / `transformer` / `vae`) and `from_pretrained`; peak memory is higher than loading overrides alone, which is acceptable for smoke checks.

`quantize.py` currently targets diffusers submodules with **torchao** and saves a standalone component directory that can be passed back into `run.py --transformer` (or another supported slot) for validation. Treat `run.py` as the acceptance-test step for a quantized component, not as the quantizer itself.

Report JSON:

```
model-path: Qwen/Qwen-Image-Edit-2511 --transformer path/to/quantized-transformer --override-weight-format pytorch --report path/to/quantized-transformer/run-report.json
```

## CLI reference

### `scripts/run.py`

| Argument | Role |
|----------|------|
| `model` | Pipeline root (relative to `MODELSCOPE_CACHE` or absolute). |
| `--unet` / `--transformer` / `--vae` | Optional local submodule directories. |
| `--torch-dtype` | `auto`, `float32`, `float16`, or `bfloat16` (default: `bfloat16`). Applied to both base pipeline and override loads. |
| `--override-weight-format` | `auto` (default), `safetensors`, or `pytorch`. Use `pytorch` for torchao quantized output (`.bin` files). `auto` detects from directory contents. |
| `--no-trust-remote-code` | Turn off `trust_remote_code`. |
| `--smoke` | After load, move pipeline to CUDA if available, else CPU. |
| `--report PATH` | Write validation metadata JSON (includes `quantization_reports` merged from each override's `quantize-report.json`). For quantized-component testing, store it inside the quantized component directory, for example `path/to/quantized-transformer/run-report.json`. |

### `scripts/quantize.py`

| Argument | Role |
|----------|------|
| `model` | Pipeline root (relative to `MODELSCOPE_CACHE` or absolute). |
| `--scheme` | `W8A8` (default, maps to `torchao int8dq`) or `W4A16` (maps to `torchao int4wo`). |
| `--target` | Component to quantize: `transformer` (default), `unet`, or `vae`. |
| `--output PATH` | Output directory for the quantized component. |
| `--group-size` | TorchAO group size for `W4A16` / `int4wo` (default: `128`). |
| `--torch-dtype` | Load dtype: `auto`, `float32`, `float16`, or `bfloat16` (default: `bfloat16`). |
| `--no-trust-remote-code` | Turn off `trust_remote_code`. |
| `--report PATH` | Write quantization metadata JSON. Default behavior is to save `quantize-report.json` inside the quantized component output directory. |

## Environment

| Variable | Default | Meaning |
|----------|---------|---------|
| `MODELSCOPE_CACHE` | `/workspace/models` | Root for relative paths |

`~/.openclaw/openclaw.json` example:

```json5
{
  skills: {
    entries: {
      imageq: {
        env: {
          MODELSCOPE_CACHE: "/workspace/models",
        },
      },
    },
  },
}
```

## Manual tmux

Attach: `tmux attach -t imageq-<short-name>` — detach with `Ctrl-b` `d`.

List: `tmux ls | grep imageq`

## Notes

- `{baseDir}/.venv` is created on the **first** tmux run on that machine (`uv venv` if missing); later runs reuse it. No repo-pinned preinstall required unless you are debugging the script.
- Isolated from **llmq** / **msdl** venvs. No `modelscope` package here, no downloads, no TensorRT stack.
- `run.py` and `quantize.py` share the same **imageq** venv. Dependencies are pinned in `requirements.txt` with minimum versions; `torch` / `torchao` CUDA wheels must match the host CUDA toolkit version.
- Overrides require diffusers-compatible submodule folders (`local_files_only=True`; weights load with the same class the base pipeline already uses for that slot).
- Current quantization backend is **torchao** only. Default mappings:
  - `W8A8` -> `torchao int8dq`
  - `W4A16` -> `torchao int4wo`
- Workflow split:
  - `quantize.py` creates the quantized component directory and writes `quantize-report.json` inside that output directory by default.
  - `run.py` validates the quantized component by mounting it back into the base pipeline; save that validation report inside the same quantized component directory, for example `run-report.json`.
- Initial validation focus is `transformer`; `unet` / `vae` support is exposed in the CLI but should be treated as less proven until tested on real checkpoints.

## When to read references

Read [references/advanced.md](references/advanced.md) for: msdl command template, ModelScope portal, torchao quantization notes, VRAM / `--smoke` OOM, submodule layout, and how to run real `pipe(...)` inference for a specific checkpoint.
