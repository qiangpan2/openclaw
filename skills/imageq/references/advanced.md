# imageq — advanced reference

## Contents

1. [msdl first](#msdl-first)
2. [ModelScope](#modelscope)
3. [TorchAO quantization](#torchao-quantization)
4. [Submodule overrides](#submodule-overrides)
5. [Dependencies](#dependencies)
6. [VRAM and smoke](#vram-and-smoke)
7. [Inference](#inference)

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
- Use the exact model ID with **msdl**'s `download.py`.

## TorchAO quantization

- `scripts/quantize.py` uses **torchao** for local diffusers submodule quantization.
- Current scheme mapping:
  - `W8A8` -> `torchao int8dq`
  - `W4A16` -> `torchao int4wo`
- Current default target is `transformer`; `unet` / `vae` are available in the CLI but should be treated as less proven until tested on real checkpoints.
- The output is a standalone diffusers component directory (`.bin` weights, `safe_serialization=False`) intended to be passed back into `scripts/run.py` via `--transformer`, `--unet`, or `--vae`.
- By default, quantization metadata should live inside that output directory as `quantize-report.json`.
- Component class discovery uses `model_index.json` to avoid loading the entire pipeline into memory. If `model_index.json` is missing or the entry cannot be resolved, `quantize.py` falls back to a full `DiffusionPipeline.from_pretrained` (high memory; a warning is printed).

## Submodule overrides

- Point `--unet`, `--transformer`, or `--vae` at a directory that contains a valid diffusers export (`config.json` + weights) for that component type.
- `run.py` itself still does not run calibration or PTQ; it loads and optionally smoke-tests.
- Quantization now lives in `scripts/quantize.py`, which prepares a compatible local component export first and then relies on `run.py` for validation.
- Treat `run.py` as the acceptance-test step for a quantized component. When validating a quantized output directory, save the run metadata back into that same directory, for example `path/to/quantized-transformer/run-report.json`.
- When loading torchao quantized overrides, pass `--override-weight-format pytorch` (or rely on `auto` if the directory only contains `.bin` files) so that `run.py` uses `use_safetensors=False`.
- Both `quantize.py` and `run.py` default to `--torch-dtype bfloat16` so that the quantization dtype and the validation dtype stay consistent.

## Dependencies

- All dependencies are declared in `requirements.txt` with minimum version constraints. Install via `uv pip install -p $VENV -r {baseDir}/requirements.txt`.
- Key version requirements:
  - `torch>=2.5` (torchao ABI compatibility)
  - `torchao>=0.7` (Int4WeightOnlyConfig, int8dq support)
  - `diffusers>=0.32` (TorchAoConfig, PipelineQuantizationConfig)
- `torch` / `torchao` CUDA wheels must match the host CUDA toolkit version. If you see segfaults or `undefined symbol` errors, the most likely cause is a torch/CUDA mismatch.
- The imageq venv is isolated from **llmq** / **msdl** venvs; do not install `modelscope` or `llmcompressor` here.

## VRAM and smoke

- `--smoke` moves the full pipeline to CUDA when available and can OOM on large checkpoints.
- If that happens, drop `--smoke` or use a smaller checkpoint / more GPU memory.

## Inference

- Arguments to `pipe(...)` depend on the pipeline class (text2img, edit, etc.). Follow the model card or repo README after a successful load.
