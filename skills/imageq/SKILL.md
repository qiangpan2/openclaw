---
name: imageq
description: "Load, smoke-test, or torchao-quantize a local diffusers image/editing pipeline from disk only (local_files_only). Use when the user mentions diffusers, image generation, image editing, Qwen-Image, DiT/transformer pipelines, loading a pipeline from disk, validating weights on GPU, submodule overrides (UNet/transformer/VAE), torchao quantization, W8A8, W4A16, or paths under /workspace/models — and the model is already on disk. Do not use for downloads or ModelScope API calls (use msdl first). Do not use for causal LLM quantization (use llmq). No TensorRT. Runs in tmux."
metadata: { "openclaw": { "emoji": "🖼️", "requires": { "bins": ["uv", "tmux"] } } }
---

# imageq — local diffusers pipelines

Run `scripts/run.py` or `scripts/quantize.py` on the **target host** in tmux. Prerequisite: pipeline files already exist (typically after **msdl**). `quantize.py` quantizes multiple pipeline components in one run and outputs a complete diffusers pipeline directory; `run.py` is the validation/acceptance step that loads the quantized pipeline and optionally moves it to GPU.

## Agent behavior

Run **imageq** in a **detached tmux session** so the chat does not block.

Start `run.py`:

```bash
tmux new-session -d -s imageq-<short-name> \
  "VENV={baseDir}/.venv && (test -d \$VENV || uv venv \$VENV) && uv pip install -p \$VENV -r {baseDir}/requirements.txt && \$VENV/bin/python {baseDir}/scripts/run.py <model-path> [flags]; exec bash"
```

Start `quantize.py`:

```bash
tmux new-session -d -s imageq-<short-name> \
  "VENV={baseDir}/.venv && (test -d \$VENV || uv venv \$VENV) && uv pip install -p \$VENV -r {baseDir}/requirements.txt && \$VENV/bin/python {baseDir}/scripts/quantize.py <model-path> [flags]; exec bash"
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

- Relative `model-path` resolves under `/workspace/models` (hardcoded).
- If the directory is missing, run **msdl** in a separate session first (`skills/msdl/SKILL.md`).

## Quick start

```
model-path: Qwen/Qwen-Image-Edit-2511
session:    imageq-qwen-image-edit
```

## Examples

Quantize transformer + vae with default `W8A8` (default targets):

```
session:    imageq-qwen-image-edit
model-path: Qwen/Qwen-Image-Edit-2511
```

Quantize only transformer with `W4A16` (`torchao int4wo`):

```
session:    imageq-qwen-image-edit
model-path: Qwen/Qwen-Image-Edit-2511 --target transformer --scheme W4A16 --group-size 128
```

List all available pipelines in `/workspace/models`:

```
session:    imageq-list
command:    run.py (no model arg)
markers:    [PIPELINE] <name> per line
```

Smoke (moves pipeline to CUDA when available):

```
session:    imageq-qwen-image-edit
model-path: Qwen/Qwen-Image-Edit-2511 --smoke
```

Validate a quantized transformer folder (single `diffusion_pytorch_model.bin` or **sharded** `diffusion_pytorch_model.bin.index.json` + `*-of-*.bin` shards):

```
model-path: Qwen/Qwen-Image-Edit-2511 --transformer path/to/quantized-transformer --override-weight-format pytorch --smoke --report path/to/quantized-transformer/run-report.json
```

With sharded PyTorch weights, `auto` also works: `run.py` detects `*.bin.index.json` and forces the PyTorch index path (`use_safetensors=False`).

Overrides load the base pipeline first, then replace each requested submodule using that component’s concrete class (`type(pipe.unet)` / `transformer` / `vae`) and `from_pretrained`; peak memory is higher than loading overrides alone, which is acceptable for smoke checks.

`quantize.py` quantizes multiple pipeline components in one run (default: `transformer vae`) and outputs a **complete diffusers pipeline directory** at `/workspace/models/<model-basename>-<scheme>/`. Non-quantized components (e.g. `text_encoder`) and small metadata dirs (`tokenizer`, `scheduler`, etc.) are fully copied. The output pipeline is self-contained with no symlinks and can be loaded directly with `DiffusionPipeline.from_pretrained`.

Report JSON:

```
model-path: Qwen/Qwen-Image-Edit-2511 --transformer path/to/quantized-transformer --override-weight-format pytorch --report path/to/quantized-transformer/run-report.json
```

## CLI reference

### `scripts/run.py`

| Argument                             | Role                                                                                                                                                                                                                                                                  |
| ------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `model`                              | Pipeline root (relative to `/workspace/models` or absolute). **Optional** — if omitted, scans `/workspace/models` and prints `[PIPELINE] <name>` for each valid pipeline, then exits.                                                                                 |
| `--unet` / `--transformer` / `--vae` | Optional local submodule directories.                                                                                                                                                                                                                                 |
| `--torch-dtype`                      | `auto`, `float32`, `float16`, or `bfloat16` (default: `bfloat16`). Applied to both base pipeline and override loads.                                                                                                                                                  |
| `--override-weight-format`           | `auto` (default), `safetensors`, or `pytorch`. Use `pytorch` for torchao quantized output (`.bin` files). `auto` detects from directory contents, including **sharded** layouts: `*.bin.index.json` / `*.safetensors.index.json` (PyTorch index forces `use_safetensors=False`). |
| `--no-trust-remote-code`             | Turn off `trust_remote_code`.                                                                                                                                                                                                                                         |
| `--smoke`                            | After load, move pipeline to CUDA if available, else CPU.                                                                                                                                                                                                             |
| `--report PATH`                      | Write validation metadata JSON (includes `quantization_reports` merged from each override's `quantize-report.json`). For quantized-component testing, store it inside the quantized component directory, for example `path/to/quantized-transformer/run-report.json`. |

### `scripts/quantize.py`

| Argument                 | Role                                                                                                                                        |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `model`                  | Pipeline root (relative to `/workspace/models` or absolute).                                                                                |
| `--scheme`               | `W8A8` (default, maps to `torchao int8dq`) or `W4A16` (maps to `torchao int4wo`).                                                           |
| `--target`               | Components to quantize (space-separated, default: `transformer vae`). Any combination of `transformer`, `unet`, `vae`.                      |
| `--output PATH`          | Output directory for the complete quantized pipeline.                                                                                       |
| `--group-size`           | TorchAO group size for `W4A16` / `int4wo` (default: `128`).                                                                                 |
| `--torch-dtype`          | Load dtype: `auto`, `float32`, `float16`, or `bfloat16` (default: `bfloat16`).                                                              |
| `--no-trust-remote-code` | Turn off `trust_remote_code`.                                                                                                               |
| `--report PATH`          | Write quantization metadata JSON. Default behavior is to save `quantize-report.json` at the pipeline root of the output pipeline directory. |

## Manual tmux

Attach: `tmux attach -t imageq-<short-name>` — detach with `Ctrl-b` `d`.

List: `tmux ls | grep imageq`

## Notes

- `{baseDir}/.venv` is created on the **first** tmux run on that machine (`uv venv` if missing); later runs reuse it. No repo-pinned preinstall required unless you are debugging the script.
- Isolated from **llmq** / **msdl** venvs. No `modelscope` package here, no downloads, no TensorRT stack.
- `run.py` and `quantize.py` share the same **imageq** venv. Dependencies are pinned in `requirements.txt` with exact pinned versions; `torch` / `torchao` CUDA wheels must match the host CUDA toolkit version.
- Overrides require diffusers-compatible submodule folders (`local_files_only=True`; weights load with the same class the base pipeline already uses for that slot).
- Current quantization backend is **torchao** only. Default mappings:
  - `W8A8` -> `torchao int8dq`
  - `W4A16` -> `torchao int4wo`
- Workflow split:
  - `quantize.py` quantizes multiple targets in one run and writes `quantize-report.json` at the **pipeline root** of the output directory by default.
  - `run.py` validates the quantized pipeline by loading it directly (`DiffusionPipeline.from_pretrained(output_path)`); save the validation report as, for example, `<output_pipeline>/run-report.json`.
- Initial validation focus is `transformer`; `unet` / `vae` support is exposed in the CLI but should be treated as less proven until tested on real checkpoints.
- `env_check.py` runs automatically at the start of both `quantize.py` and `run.py`. Agent output parsing: `[ENV OK]` = environment is compatible and quantization will proceed; `[ENV ERROR]` = version mismatch detected, fix command is printed on the next line — reinstall the venv with `uv pip install -p {baseDir}/.venv -r {baseDir}/requirements.txt --reinstall`.
- `run.py` with no model argument scans `/workspace/models` and prints `[PIPELINE] <name>` for each valid diffusers pipeline found (one per line), then exits 0. Agent parsing: collect all `[PIPELINE]` lines to discover available models before calling `run.py <name>` to validate a specific one.

## When to read references

Read [references/advanced.md](references/advanced.md) for: msdl command template, ModelScope portal, torchao quantization notes, VRAM / `--smoke` OOM, submodule layout, and how to run real `pipe(...)` inference for a specific checkpoint.
