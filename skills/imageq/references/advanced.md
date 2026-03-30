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
  "uv run {msdlBase}/scripts/download.py Qwen/Qwen-Image-Edit-2511"
```

Then use **imageq** with model path `Qwen/Qwen-Image-Edit-2511` relative to `/workspace/models`.

## ModelScope

- Site: [modelscope.cn](https://www.modelscope.cn)
- Use the exact model ID with **msdl**'s `download.py`.

## TorchAO quantization

- `scripts/quantize.py` uses **torchao** for local diffusers pipeline quantization.
- Current scheme mapping:
  - `W8A8` -> `torchao int8dq`
  - `W4A16` -> `torchao int4wo`
- Default targets: `transformer vae`. Pass `--target transformer` to quantize only one component.
- Output is a **complete diffusers pipeline directory** at `/workspace/models/<model-basename>-<scheme>/`:
  - Quantized components: saved with `.bin` weights (`safe_serialization=False`; large checkpoints may be **sharded** with `*.bin.index.json` and `*-of-*.bin` shard files)
  - Non-quantized large components (e.g. `text_encoder`): fully copied
  - Small metadata dirs (`tokenizer`, `scheduler`, `processor`): copied
  - Root `quantize-report.json`: records all quantized components and scheme
- The output pipeline can be loaded directly with `DiffusionPipeline.from_pretrained(output_path, local_files_only=True)`.
- **Copy note:** non-quantized pipeline components (e.g. `text_encoder`) are fully copied into the output directory. No symlinks are used; the output pipeline directory is self-contained and can be moved independently.
- Report format (`quantize-report.json` at pipeline root):
  ```json
  {
    "source_pipeline": "/workspace/models/Qwen/Qwen-Image-Edit-2511",
    "output_pipeline": "/workspace/models/Qwen-Image-Edit-2511-W8A8",
    "scheme": "W8A8",
    "torchao_scheme": "int8dq",
    "torch_dtype": "bfloat16",
    "group_size": null,
    "quantized_at": "2026-03-26T00:00:00+00:00",
    "components": {
      "transformer": { "class": "Qwen2_5OmniTransformerModel", "source": "..." },
      "vae": { "class": "AutoencoderKL", "source": "..." }
    }
  }
  ```
- Component class discovery uses `model_index.json` to avoid loading the entire pipeline into memory. If `model_index.json` is missing or the entry cannot be resolved, `quantize.py` falls back to a full `DiffusionPipeline.from_pretrained` (high memory; a warning is printed).

## Submodule overrides (run.py)

- Point `--unet`, `--transformer`, or `--vae` at a directory that contains a valid diffusers export (`config.json` + weights) for that component type.
- Supported weight layouts for overrides include:
  - single-file `.safetensors` or `.bin`
  - **sharded safetensors**: `*.safetensors.index.json` + shard `*.safetensors`
  - **sharded PyTorch**: `*.bin.index.json` + shard `*.bin` (e.g. `diffusion_pytorch_model-00001-of-00003.bin`)
- In `--override-weight-format auto`, `run.py` checks for `*.bin.index.json` / `*.safetensors.index.json` **before** inferring from loose `*.bin` / `*.safetensors`, so sharded PyTorch exports are not misclassified.
- `run.py` without a model argument scans `/workspace/models` and prints `[PIPELINE] <name>` for each valid pipeline (contains `model_index.json`), then exits. Use this to discover available quantized pipelines before calling `run.py <name>`.
- **Two valid workflows:** (1) After `quantize.py`, load the **full output pipeline** path as `model` (simplest). (2) Load the original pipeline as `model` and pass only a replaced component directory via `--transformer` / `--vae` / `--unet` (e.g. to test one sharded or quantized folder).
- When validating torchao quantized overrides explicitly, pass `--override-weight-format pytorch` or use `auto` (recommended for sharded dirs: `auto` detects the index file).
- Both `quantize.py` and `run.py` default to `--torch-dtype bfloat16` so that the quantization dtype and the validation dtype stay consistent.

## Dependencies

- All dependencies are declared in `requirements.txt` with **exact pinned versions** (`==`) and a torch-specific CUDA index header. Install via `uv pip install -p $VENV -r {baseDir}/requirements.txt`.
- Key version requirements:
  - `torch+cu126` (CUDA 12.x build — system driver is CUDA 12.9, cu130 builds are incompatible)
  - `torchao` pinned to a version where `float8_dynamic_activation_float8_weight` is available and ABI-compatible with the pinned torch
  - `diffusers` pinned to a version whose `TorchAoConfig` is compatible with the pinned torchao
- `torch` / `torchao` CUDA wheels must match the host CUDA toolkit version. If you see segfaults or `undefined symbol` errors, the most likely cause is a torch/CUDA mismatch.
- The imageq venv is isolated from **llmq** / **msdl** venvs; do not install `modelscope` or `llmcompressor` here.

## 已验证版本组合

| 验证日期   | torch       | torchvision  | torchao | diffusers | CUDA build | 状态 |
| ---------- | ----------- | ------------ | ------- | --------- | ---------- | ---- |
| 2026-03-26 | 2.7.1+cu126 | 0.22.1+cu126 | 0.15.0  | 0.37.1    | cu126      | ✓    |

## 依赖升级流程（agent 执行）

触发条件：新模型需要更新 diffusers/torchao，或 tmux 里出现 `[ENV ERROR]`。

```bash
# 1. 查候选版本
uv pip index versions torch --extra-index-url https://download.pytorch.org/whl/cu126
uv pip index versions torchao
uv pip index versions diffusers

# 2. 临时 venv 测试候选组合（替换实际版本号）
uv venv /tmp/imageq-test-venv --python 3.11
uv pip install -p /tmp/imageq-test-venv \
  "torch==${NEW_TORCH}+cu126" "torchvision==${NEW_TV}+cu126" \
  "torchao==${NEW_TORCHAO}" "diffusers==${NEW_DIFFUSERS}" \
  "transformers>=4.38" "accelerate>=0.26" "safetensors>=0.4" \
  --extra-index-url https://download.pytorch.org/whl/cu126
/tmp/imageq-test-venv/bin/python skills/imageq/scripts/env_check.py

# 3a. [ENV OK] → 更新 requirements.txt + 版本兼容表 → commit
# 3b. [ENV ERROR] → 降版本重试步骤 2
rm -rf /tmp/imageq-test-venv
```

## VRAM and smoke

- `--smoke` moves the full pipeline to CUDA when available and can OOM on large checkpoints.
- If that happens, drop `--smoke` or use a smaller checkpoint / more GPU memory.

## Inference

- Arguments to `pipe(...)` depend on the pipeline class (text2img, edit, etc.). Follow the model card or repo README after a successful load.
