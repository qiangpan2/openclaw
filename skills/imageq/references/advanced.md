# imageq — advanced reference

## Contents

1. [msdl first](#msdl-first)
2. [ModelScope](#modelscope)
3. [Submodule overrides](#submodule-overrides)
4. [VRAM and smoke](#vram-and-smoke)
5. [Inference](#inference)

## msdl first

**msdl** downloads; **imageq** only reads local files. If the path is missing, start **msdl** in its own tmux session (see `skills/msdl/SKILL.md`).

Example (replace `{msdlBase}` with the msdl skill directory on the host):

```bash
tmux new-session -d -s ms-dl-qwen-image-edit \
  "MODELSCOPE_CACHE=/workspace/models uv run {msdlBase}/scripts/download.py Qwen/Qwen-Image-Edit-2511"
```

Then use **imageq** with model path `Qwen/Qwen-Image-Edit-2511` relative to `MODELSCOPE_CACHE`.

## ModelScope

- Site: [modelscope.cn](https://www.modelscope.cn)
- Use the exact model ID with **msdl**’s `download.py`.

## Submodule overrides

- Point `--unet`, `--transformer`, or `--vae` at a directory that contains a valid diffusers export (`config.json` + weights) for that component type.
- **imageq** v1 does not run calibration or full PTQ; it loads and optionally smoke-tests.

## VRAM and smoke

- `--smoke` moves the full pipeline to CUDA when available and can OOM on large checkpoints.
- If that happens, drop `--smoke` or use a smaller checkpoint / more GPU memory.

## Inference

- Arguments to `pipe(...)` depend on the pipeline class (text2img, edit, etc.). Follow the model card or repo README after a successful load.
