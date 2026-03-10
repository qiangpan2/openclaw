---
name: msdl
description: "Download models from ModelScope (魔搭). Use when user wants to download a model by model ID (e.g. stepfun-ai/Step-3.5-Flash, Qwen/Qwen3-32B) from modelscope.cn to a local directory. No login required for public models. Supports filtering files by glob pattern and specifying revisions."
metadata:
  {
    "openclaw":
      {
        "emoji": "📦",
        "requires": { "bins": ["uv", "tmux"] },
      },
  }
---

# ModelScope Download

Download models from [ModelScope](https://www.modelscope.cn) using `uv run` inside a tmux detached session. Downloads run in the background without blocking the conversation.

## Agent behavior

IMPORTANT: Always run downloads inside a tmux detached session to avoid blocking the conversation.

Start download:

```bash
tmux new-session -d -s ms-dl-<short-name> \
  "MODELSCOPE_CACHE=/workspace/models uv run {baseDir}/scripts/download.py <model-id> [flags]; exec bash"
```

Session naming: use `ms-dl-` prefix + lowercase model short name (e.g. `ms-dl-step-3.5-flash`, `ms-dl-qwen3-32b`).

After starting, report the tmux session name to the user and stop. Do NOT poll or monitor progress automatically.

Check progress (only when user asks):

```bash
tmux capture-pane -t ms-dl-<short-name> -p | tail -50
```

Check if download is complete: look for `Done. Model saved to:` in the captured output.

Get full log:

```bash
tmux capture-pane -t ms-dl-<short-name> -p -S -
```

## Quick start

```bash
tmux new-session -d -s ms-dl-step-3.5-flash \
  "MODELSCOPE_CACHE=/workspace/models uv run {baseDir}/scripts/download.py stepfun-ai/Step-3.5-Flash; exec bash"
```

## Examples

```bash
tmux new-session -d -s ms-dl-qwen3-32b \
  "MODELSCOPE_CACHE=/workspace/models uv run {baseDir}/scripts/download.py Qwen/Qwen3-32B --include '*.safetensors'; exec bash"

tmux new-session -d -s ms-dl-wan2.2 \
  "uv run {baseDir}/scripts/download.py Wan-AI/Wan2.2-T2V-A14B --local-dir ./my-models; exec bash"
```

## Environment

| Variable | Default | Description |
|----------|---------|-------------|
| `MODELSCOPE_CACHE` | `/workspace/models` | Default cache/download directory |

Or configure in `~/.openclaw/openclaw.json`:

```json5
{
  skills: {
    entries: {
      "msdl": {
        env: {
          MODELSCOPE_CACHE: "/workspace/models",
        },
      },
    },
  },
}
```

## Manual access

Attach to a running download session to see real-time output:

```bash
tmux attach -t ms-dl-<short-name>
```

Detach without stopping: press `Ctrl-b` then `d`.

List all download sessions:

```bash
tmux ls | grep ms-dl
```

## Notes

- Public models require no login or token.
- Private/gated models need `MODELSCOPE_TOKEN` set, or run `modelscope login` first.
- `exec bash` at the end keeps the tmux session alive after download completes, so you can still inspect the output.
- Re-running a download for the same model skips already completed files (resume-safe).
