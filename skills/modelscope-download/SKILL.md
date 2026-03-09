---
name: modelscope-download
description: "Download models from ModelScope (魔搭). Use when user wants to download a model by model ID (e.g. stepfun-ai/Step-3.5-Flash, Qwen/Qwen3-32B) from modelscope.cn to a local directory. No login required for public models. Supports filtering files by glob pattern and specifying revisions."
metadata:
  {
    "openclaw":
      {
        "emoji": "📦",
        "requires": { "bins": ["uv"] },
      },
  }
---

# ModelScope Download

Download models from [ModelScope](https://www.modelscope.cn) using `uv run` (auto-installs dependencies in an isolated environment).

## Quick start

```bash
uv run {baseDir}/scripts/download.py stepfun-ai/Step-3.5-Flash
```

Models are saved to `/workspace/models` by default. Override with `MODELSCOPE_CACHE` or `--local-dir`.

## Useful flags

```bash
uv run {baseDir}/scripts/download.py Qwen/Qwen3-32B --include "*.safetensors"
uv run {baseDir}/scripts/download.py Wan-AI/Wan2.2-T2V-A14B --local-dir ./my-models
uv run {baseDir}/scripts/download.py stepfun-ai/Step-3.5-Flash --revision main --exclude "*.bin"
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
      "modelscope-download": {
        env: {
          MODELSCOPE_CACHE: "/workspace/models",
        },
      },
    },
  },
}
```

## Notes

- Public models require no login or token.
- Private/gated models need `MODELSCOPE_TOKEN` set, or run `modelscope login` first.
