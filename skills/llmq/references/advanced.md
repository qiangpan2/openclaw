# INT8/INT4 Quantization — Advanced Guide

## Contents

- [Calibration samples](#calibration-samples)
- [Smoothing strength (W8A8 only)](#smoothing-strength-w8a8-only)
- [Scheme comparison](#scheme-comparison-w8a8-vs-w8a16-vs-w4a16)
- [Group size (W4A16 only)](#group-size-w4a16-only)
- [Custom calibration datasets](#custom-calibration-datasets)
- [Model-specific notes](#model-specific-notes)
- [Verifying with vLLM](#verifying-with-vllm)
- [Inspecting the quantization report](#inspecting-the-quantization-report)

## Calibration samples

The `--samples` parameter controls how many examples from the calibration dataset are used to estimate quantization scales. More samples generally improve accuracy but increase quantization time.

| Model size | Recommended `--samples` |
|------------|------------------------|
| < 3B       | 512 – 1024             |
| 3B – 13B   | 256 – 512              |
| > 13B      | 128 – 256              |

## Smoothing strength (W8A8 only)

`--smoothing` controls the SmoothQuant migration factor. It balances quantization difficulty between weights and activations:

- **0.5** — balanced between weights and activations; safer for models with extreme activation outliers
- **0.8** — default; works well for most transformer architectures
- **0.9** — shifts more difficulty to weights; can help when activation distributions are highly uniform

If quantized model quality is poor, try values between 0.5 and 0.9 in increments of 0.1.

## Scheme comparison: W8A8 vs W8A16 vs W4A16

| Aspect      | W8A8                                   | W8A16                          | W4A16                                  |
|-------------|----------------------------------------|--------------------------------|----------------------------------------|
| Weights     | INT8                                   | INT8                           | INT4 (group quantized)                 |
| Activations | INT8 (dynamic per-token at inference)  | FP16 (unchanged)               | FP16 (unchanged)                       |
| Compute     | INT8 matmul                            | FP16 matmul                    | FP16 matmul (with INT4 dequant)        |
| Model size  | ~50% of FP16                           | ~50% of FP16                   | ~25% of FP16                           |
| Speed gain  | Significant (high QPS / large batch)   | Modest (mainly memory saving)  | Modest (mainly memory saving)          |
| Accuracy    | Slightly lower                         | Closer to original             | Lower than INT8; group_size helps      |
| SmoothQuant | Yes                                    | Not used                       | Not used                               |
| Group quant | Not used                               | Not used                       | Yes (default group_size=128)           |

## Group size (W4A16 only)

`--group-size` controls the granularity of INT4 quantization. Weights are divided into groups of this size, and each group gets its own scale/zero-point. Smaller groups preserve more precision but increase metadata overhead.

| Group size | Trade-off                                                        |
|------------|------------------------------------------------------------------|
| **64**     | Best accuracy; higher memory overhead from more scale parameters |
| **128**    | Default; good balance between accuracy and efficiency            |
| **256**    | Smallest overhead; may degrade accuracy on sensitive models      |

If W4A16 quantized model quality is poor, try reducing group_size from 128 to 64.

## Custom calibration datasets

The default dataset (`HuggingFaceH4/ultrachat_200k`) works well for general-purpose chat models. For domain-specific models, provide a HuggingFace dataset ID with a `train_sft` split containing a `messages` column:

```bash
--dataset "your-org/your-dataset"
```

The dataset must have a `messages` field compatible with `tokenizer.apply_chat_template()`.

## Model-specific notes

### Qwen / Qwen2 / Qwen3

- Works well with default settings.
- For Qwen models with large vocab (150K+), quantization may use more memory. Ensure sufficient GPU RAM or use `--samples 256`.

### LLaMA / LLaMA 3

- Default smoothing (0.8) is suitable.
- LLaMA 3 models with GQA work correctly with the default `targets="Linear"`.

### Mistral / Mixtral

- Mixtral (MoE) models: all expert Linear layers are quantized by default. This is correct behavior.
- If memory is tight during calibration, reduce `--samples`.

## Verifying with vLLM

After quantization, load the model in vLLM to verify it works. llmcompressor saves all schemes (W8A8, W8A16, W4A16) in compressed-tensors format, and vLLM auto-detects it from the model's `config.json`, so `--quantization` can usually be omitted:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model /workspace/models/Qwen/Qwen3.5-0.8B-W8A8 \
  --dtype auto
```

If auto-detection fails, explicitly pass `--quantization compressed-tensors`:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model /workspace/models/Qwen/Qwen3-32B-W4A16 \
  --dtype auto \
  --quantization compressed-tensors
```

Quick smoke test (works for all schemes):

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_tokens": 50}'
```

## Inspecting the quantization report

Each quantized model directory contains a `quantize-report.json` with all parameters used:

```bash
cat /workspace/models/Qwen/Qwen3.5-0.8B-W8A8/quantize-report.json
```

This is useful for reproducing quantization or comparing different configurations.
