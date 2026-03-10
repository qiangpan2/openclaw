---
name: int8q
description: "Quantize a local model to INT8 using llmcompressor. Supports W8A8 (weight+activation INT8) and W8A16 (weight INT8, activation FP16) schemes. Use when user wants to quantize, compress, or convert a model to INT8 format. Requires a local model path (e.g. from msdl download). Runs in tmux background session."
metadata:
  {
    "openclaw":
      {
        "emoji": "🔧",
        "requires": { "bins": ["uv", "tmux"] },
      },
  }
---

# INT8 Quantization

Quantize a local model to INT8 (W8A8 or W8A16) using [llmcompressor](https://github.com/vllm-project/llm-compressor) inside a tmux detached session. Quantization runs in the background without blocking the conversation.

## Agent behavior

IMPORTANT: Always run quantization inside a tmux detached session to avoid blocking the conversation.

Start quantization:

```bash
tmux new-session -d -s int8q-<short-name> \
  "VENV={baseDir}/.venv && (test -d \$VENV || uv venv \$VENV) && uv pip install -p \$VENV --upgrade llmcompressor datasets sentencepiece && \$VENV/bin/python {baseDir}/scripts/quantize.py <model-path> [flags]; exec bash"
```

Session naming: use `int8q-` prefix + lowercase model short name (e.g. `int8q-qwen3.5-0.8b`, `int8q-llama3-8b`).

After starting, report the tmux session name to the user and stop. Do NOT poll or monitor progress automatically.

Check progress (only when user asks):

```bash
tmux capture-pane -t int8q-<short-name> -p | tail -50
```

Check if quantization is complete: look for `Done. Quantized model saved to:` in the captured output.

Get full log:

```bash
tmux capture-pane -t int8q-<short-name> -p -S -
```

## Quick start

```bash
tmux new-session -d -s int8q-qwen3.5-0.8b \
  "VENV={baseDir}/.venv && (test -d \$VENV || uv venv \$VENV) && uv pip install -p \$VENV --upgrade llmcompressor datasets sentencepiece && \$VENV/bin/python {baseDir}/scripts/quantize.py /workspace/models/Qwen/Qwen3.5-0.8B; exec bash"
```

## Examples

W8A16 quantization (weight-only INT8, keeps activations in FP16):

```bash
tmux new-session -d -s int8q-llama3-8b \
  "VENV={baseDir}/.venv && (test -d \$VENV || uv venv \$VENV) && uv pip install -p \$VENV --upgrade llmcompressor datasets sentencepiece && \$VENV/bin/python {baseDir}/scripts/quantize.py /workspace/models/meta-llama/Llama-3-8B --scheme W8A16; exec bash"
```

Custom calibration settings:

```bash
tmux new-session -d -s int8q-qwen3-32b \
  "VENV={baseDir}/.venv && (test -d \$VENV || uv venv \$VENV) && uv pip install -p \$VENV --upgrade llmcompressor datasets sentencepiece && \$VENV/bin/python {baseDir}/scripts/quantize.py /workspace/models/Qwen/Qwen3-32B --samples 256 --smoothing 0.7; exec bash"
```

## Parameters

| Parameter     | Default                        | Description                                             |
|---------------|--------------------------------|---------------------------------------------------------|
| `model`       | (required)                     | Local model path                                        |
| `--scheme`    | `W8A8`                         | `W8A8` (weight+activation INT8) or `W8A16` (weight INT8, activation FP16) |
| `--output`    | `<model>-<scheme>`             | Output directory                                        |
| `--samples`   | `512`                          | Number of calibration samples                           |
| `--seq-len`   | `2048`                         | Max calibration sequence length                         |
| `--dataset`   | `HuggingFaceH4/ultrachat_200k` | Calibration dataset                                    |
| `--smoothing` | `0.8`                          | SmoothQuant strength (W8A8 only)                        |

## Environment

| Variable | Default | Description |
|----------|---------|-------------|
| `MODELSCOPE_CACHE` | `/workspace/models` | Default directory for locating source models |

Or configure in `~/.openclaw/openclaw.json`:

```json5
{
  skills: {
    entries: {
      "int8q": {
        env: {
          MODELSCOPE_CACHE: "/workspace/models",
        },
      },
    },
  },
}
```

## Manual access

Attach to a running quantization session to see real-time output:

```bash
tmux attach -t int8q-<short-name>
```

Detach without stopping: press `Ctrl-b` then `d`.

List all quantization sessions:

```bash
tmux ls | grep int8q
```

## Notes

- Dependencies (llmcompressor, datasets, sentencepiece and their transitive deps like torch, transformers) are installed into a fully isolated venv at `{baseDir}/.venv/`, with no system package inheritance.
- The venv is created on first run and reused on subsequent runs.
- To force a clean reinstall of dependencies, delete the `.venv/` directory.

## Advanced

For calibration tuning, model-specific notes, and vLLM verification, see [references/advanced.md](references/advanced.md).
