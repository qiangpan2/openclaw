---
name: llmq
description: "Quantize a local causal LLM with llmcompressor (W8A8 default, W8A16, W4A16). Use when the user asks to quantize, compress, or export an LLM to INT8 or INT4, mentions SmoothQuant, GPTQ, W4A16, calibration, llmcompressor, vLLM compressed-tensors, or model paths under MODELSCOPE_CACHE for language models. Not for diffusers or image pipelines (use imageq). Requires weights already on disk (e.g. after msdl). Runs in tmux."
metadata: { "openclaw": { "emoji": "🔧", "requires": { "bins": ["uv", "tmux"] } } }
---

# llmq — LLM quantization

Run `scripts/quantize.py` via the tmux template below. Uses [llmcompressor](https://github.com/vllm-project/llm-compressor).

## Agent behavior

Always use a **detached tmux session**.

Start:

```bash
tmux new-session -d -s llmq-<short-name> \
  "VENV={baseDir}/.venv && (test -d \$VENV || uv venv \$VENV) && uv pip install -p \$VENV --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple -r {baseDir}/requirements.txt && \$VENV/bin/python {baseDir}/scripts/quantize.py <model-path> [flags]; exec bash"
```

- Session name: `llmq-` + short model slug (e.g. `llmq-qwen3.5-0.8b`).
- After starting, report the session name and stop. Do not poll unless the user asks.

Progress:

```bash
tmux capture-pane -t llmq-<short-name> -p | tail -50
```

Done when the log shows `Done. Quantized model saved to:`.

Full log:

```bash
tmux capture-pane -t llmq-<short-name> -p -S -
```

## Paths

Relative `<model-path>` resolves under `MODELSCOPE_CACHE` (default `/workspace/models`), same as **msdl**.

## Quick start

```
model-path: Qwen/Qwen3.5-0.8B
session:    llmq-qwen3.5-0.8b
```

## Examples

Same tmux template; only path, session, and flags change.

INT4 (W4A16):

```
session:    llmq-qwen3-32b
model-path: Qwen/Qwen3-32B --scheme W4A16
```

Weight-only INT8 (W8A16):

```
session:    llmq-llama3-8b
model-path: meta-llama/Llama-3-8B --scheme W8A16
```

INT4 with tighter groups:

```
session:    llmq-llama3-70b
model-path: meta-llama/Llama-3-70B --scheme W4A16 --group-size 64 --samples 256
```

## Parameters

| Parameter      | Default                        | Description                                     |
| -------------- | ------------------------------ | ----------------------------------------------- |
| `model`        | required                       | Path relative to `MODELSCOPE_CACHE` or absolute |
| `--scheme`     | `W8A8`                         | `W8A8`, `W4A16`, or `W8A16`                     |
| `--output`     | `<model>-<scheme>`             | Output directory                                |
| `--samples`    | `512`                          | Calibration sample count                        |
| `--seq-len`    | `2048`                         | Max calibration sequence length                 |
| `--dataset`    | `HuggingFaceH4/ultrachat_200k` | HF dataset id                                   |
| `--smoothing`  | `0.8`                          | SmoothQuant strength (W8A8 only)                |
| `--group-size` | `128`                          | Group size (W4A16 only)                         |

## Environment

| Variable           | Default             | Description                   |
| ------------------ | ------------------- | ----------------------------- |
| `MODELSCOPE_CACHE` | `/workspace/models` | Base for relative model paths |

`~/.openclaw/openclaw.json`:

```json5
{
  skills: {
    entries: {
      llmq: {
        env: {
          MODELSCOPE_CACHE: "/workspace/models",
        },
      },
    },
  },
}
```

## Manual tmux

Attach: `tmux attach -t llmq-<short-name>` — detach: `Ctrl-b` then `d`.

List: `tmux ls | grep llmq`

## Notes

- `{baseDir}/.venv` is created on the **first** run on that host if missing; reuse afterward. Delete `.venv/` to force a clean dependency reinstall.
- Venv is isolated from other skills (e.g. **imageq**, **msdl**).
- To upgrade dependencies: edit `requirements.txt` with new version numbers, delete `.venv/`, then re-run. When upgrading `llmcompressor`, verify its declared torch range first (check PyPI) and update `torch`/`torchvision` pins in sync if needed.

## When to read references

Open [references/advanced.md](references/advanced.md) for calibration sizing, scheme tradeoffs, custom datasets, per-family notes, and vLLM smoke tests.
