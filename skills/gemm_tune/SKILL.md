---
name: gemm_tune
homepage: https://github.com/AMD-CSE-Radeon-AI/AutoTune
description: "AutoTune is a Python-driven workflow that profiles GEMM workloads, extracts problem sizes from hipblaslt-bench logs, then regenerates hipBLASLt/Tensile configuration."
metadata:
  {
    "openclaw":
      {
        "emoji": "🚀",
        "requires": { "bins": ["python3", "git"] },
      },
  }
---

# AutoTune (hipBLASLt GEMM Auto-Tuning)

AutoTune is a Python-driven workflow that profiles GEMM workloads, extracts problem sizes from `hipblaslt-bench` logs, then regenerates hipBLASLt/Tensile configuration and runs a second benchmark pass to measure improvements.

The python script is retrieved from the AutoTune repository:
[https://github.com/AMD-CSE-Radeon-AI/AutoTune](https://github.com/AMD-CSE-Radeon-AI/AutoTune)

## Quick start

```bash
[ ! -d "AutoTune" ] && git clone https://github.com/AMD-CSE-Radeon-AI/AutoTune.git
cd AutoTune
python3 tune_file.py -f <path_or_url_to_log>
```

## Examples

```bash
[ ! -d "AutoTune" ] && git clone https://github.com/AMD-CSE-Radeon-AI/AutoTune.git
cd AutoTune

# Provide a local log path:
python3 tune_file.py -f /path/to/local/ComfyUI.log

# Or provide a web URL:
python3 tune_file.py -f https://example.com/logs/ComfyUI.log
```

## Environment

This project is tailored to run in a **ROCm + PyTorch environment**. 
Key environment variables used by AutoTune:

- `GPU_TARGET=<gfx...>` (detected via `rocminfo`)
- `TORCH_BLAS_PREFER_HIPBLASLT=1`
- `ROCBLAS_USE_HIPBLASLT=1` (llama.cpp path)
- `HIPBLASLT_LOG_MASK=32`
- `HIPBLASLT_LOG_FILE=hipblaslt-bench.log`
- `HIPBLASLT_TUNING_OVERRIDE_FILE=.../QuickTune/test_tuning`

## Manual access

Run the tuning script directly:
```bash
[ ! -d "AutoTune" ] && git clone https://github.com/AMD-CSE-Radeon-AI/AutoTune.git
cd AutoTune
python3 tune_file.py -f <path_or_url_to_log>
```
