---
name: imageq
description: "Load or smoke-test a diffusers image/editing pipeline from a local directory only (local_files_only). Use when the user mentions diffusers, image generation, image editing, Qwen-Image, DiT/transformer pipelines, loading a pipeline from disk, validating weights on GPU, submodule overrides (UNet/transformer/VAE), or paths under MODELSCOPE_CACHE — and the model is already on disk. Do not use for downloads or ModelScope API calls (use msdl first). Do not use for causal LLM quantization (use llmq). No TensorRT. Runs in tmux."
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

Run `scripts/run.py` on the **target host** in tmux. Prerequisite: pipeline files already exist (typically after **msdl**).

## Agent behavior

Run **imageq** in a **detached tmux session** so the chat does not block.

Start:

```bash
tmux new-session -d -s imageq-<short-name> \
  "VENV={baseDir}/.venv && (test -d \$VENV || uv venv \$VENV) && uv pip install -p \$VENV --upgrade diffusers transformers accelerate safetensors torch torchvision && MODELSCOPE_CACHE=\${MODELSCOPE_CACHE:-/workspace/models} \$VENV/bin/python {baseDir}/scripts/run.py <model-path> [flags]; exec bash"
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

Smoke (moves pipeline to CUDA when available):

```
session:    imageq-qwen-image-edit
model-path: Qwen/Qwen-Image-Edit-2511 --smoke
```

Override a local transformer folder:

```
model-path: Qwen/Qwen-Image-Edit-2511 --transformer path/to/quantized-transformer --smoke
```

Overrides load the base pipeline first, then replace each requested submodule using that component’s concrete class (`type(pipe.unet)` / `transformer` / `vae`) and `from_pretrained`; peak memory is higher than loading overrides alone, which is acceptable for smoke checks.

Report JSON:

```
model-path: Qwen/Qwen-Image-Edit-2511 --report /tmp/imageq-report.json
```

## CLI reference

| Argument | Role |
|----------|------|
| `model` | Pipeline root (relative to `MODELSCOPE_CACHE` or absolute). |
| `--unet` / `--transformer` / `--vae` | Optional local submodule directories. |
| `--no-trust-remote-code` | Turn off `trust_remote_code`. |
| `--smoke` | After load, move pipeline to CUDA if available, else CPU. |
| `--report PATH` | Write run metadata JSON. |

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
- Overrides require diffusers-compatible submodule folders (`local_files_only=True`; weights load with the same class the base pipeline already uses for that slot).

## When to read references

Read [references/advanced.md](references/advanced.md) for: msdl command template, ModelScope portal, VRAM / `--smoke` OOM, submodule layout, and how to run real `pipe(...)` inference for a specific checkpoint.
