"""Load a local diffusers pipeline from disk (ModelScope layout via msdl). No downloads."""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from paths import resolve_model_path


def _parse_torch_dtype(raw: str):
    import torch

    mapping = {
        "auto": None,
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[raw]


def _detect_safetensors(override_dir: Path, mode: str):
    """Return the use_safetensors kwarg value based on mode and directory contents."""
    if mode == "safetensors":
        return True
    if mode == "pytorch":
        return False
    # auto: peek at what's in the directory
    has_st = any(override_dir.glob("*.safetensors"))
    has_bin = any(override_dir.glob("*.bin"))
    if has_st:
        return True
    if has_bin:
        return False
    return None  # let diffusers decide


def replace_submodule(pipe, attr: str, path: Path, trust_remote_code: bool,
                      label: str, torch_dtype=None, use_safetensors=None) -> None:
    """Load weights with the same concrete class as the pipeline's existing submodule."""
    sub = getattr(pipe, attr, None)
    if sub is None:
        print(
            f"Error: this pipeline has no {label} to override (missing or None attribute `{attr}`).",
            file=sys.stderr,
        )
        sys.exit(1)
    cls = type(sub)
    load_kw = {
        "local_files_only": True,
        "trust_remote_code": trust_remote_code,
    }
    if torch_dtype is not None:
        load_kw["torch_dtype"] = torch_dtype
    if use_safetensors is not None:
        load_kw["use_safetensors"] = use_safetensors
    replacement = cls.from_pretrained(str(path), **load_kw)
    setattr(pipe, attr, replacement)


def _read_quantize_report(override_dir: Path):
    """Read quantize-report.json from an override directory, if it exists."""
    rpt = override_dir / "quantize-report.json"
    if not rpt.is_file():
        return None
    try:
        with open(rpt, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Load a local diffusers pipeline (no download; use msdl first)"
    )
    parser.add_argument(
        "model",
        help="Pipeline directory: relative to MODELSCOPE_CACHE or absolute",
    )
    parser.add_argument(
        "--unet",
        default=None,
        help="Optional local directory to override UNet weights",
    )
    parser.add_argument(
        "--transformer",
        default=None,
        help="Optional local directory to override transformer (e.g. DiT) weights",
    )
    parser.add_argument(
        "--vae",
        default=None,
        help="Optional local directory to override VAE weights",
    )
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="bfloat16",
        help="torch_dtype for pipeline and override loads (default: bfloat16)",
    )
    parser.add_argument(
        "--override-weight-format",
        choices=["auto", "safetensors", "pytorch"],
        default="auto",
        help="Weight format for override directories: auto detects from files, "
             "pytorch forces .bin (for torchao quantized output), safetensors forces .safetensors (default: auto)",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass trust_remote_code to from_pretrained (default: true)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="After load, move pipeline to CUDA if available (else CPU) to validate tensors",
    )
    parser.add_argument(
        "--report",
        default=None,
        metavar="PATH",
        help="Write JSON report to this path (default: skip)",
    )
    args = parser.parse_args()

    base = resolve_model_path(args.model)
    torch_dtype = _parse_torch_dtype(args.torch_dtype)

    print(f"Model dir: {base}")
    print(f"MODELSCOPE_CACHE: {os.environ.get('MODELSCOPE_CACHE', '/workspace/models')}")
    print(f"torch_dtype: {args.torch_dtype}")
    print(f"override_weight_format: {args.override_weight_format}")
    print()

    from diffusers import DiffusionPipeline

    load_kw = {
        "local_files_only": True,
        "trust_remote_code": args.trust_remote_code,
    }
    if torch_dtype is not None:
        load_kw["torch_dtype"] = torch_dtype
    print("Loading DiffusionPipeline ...")
    pipe = DiffusionPipeline.from_pretrained(str(base), **load_kw)
    print("Load OK.")

    overrides = {}
    quantization_reports = {}

    for attr, arg_val, label in [
        ("unet", args.unet, "UNet"),
        ("transformer", args.transformer, "transformer"),
        ("vae", args.vae, "VAE"),
    ]:
        if arg_val is None:
            continue
        op = resolve_model_path(arg_val)
        use_st = _detect_safetensors(op, args.override_weight_format)
        print(f"Loading override {label} from {op} (use_safetensors={use_st}) ...")
        replace_submodule(pipe, attr, op, args.trust_remote_code, label,
                          torch_dtype=torch_dtype, use_safetensors=use_st)
        overrides[attr] = str(op)
        print(f"{label} override OK.")

        qr = _read_quantize_report(op)
        if qr is not None:
            quantization_reports[attr] = qr

    if args.smoke:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Smoke: moving pipeline to {device} ...")
        pipe.to(device)
        print("Smoke OK.")

    if args.report:
        report = {
            "pipeline_path": str(base),
            "loaded_at": datetime.now(timezone.utc).isoformat(),
            "local_files_only": True,
            "trust_remote_code": args.trust_remote_code,
            "torch_dtype": args.torch_dtype,
            "override_weight_format": args.override_weight_format,
            "overrides": overrides,
            "quantization_reports": quantization_reports if quantization_reports else None,
            "smoke_ran": args.smoke,
        }
        out = Path(args.report)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Report written to {out}")

    print("\nDone.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
