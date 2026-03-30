"""Load a local diffusers pipeline from disk (ModelScope layout via msdl). No downloads."""

import argparse
import inspect
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from paths import resolve_model_path, MODELS_ROOT
from env_check import check_env


def _parse_torch_dtype(raw: str):
    import torch

    mapping = {
        "auto": None,
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[raw]


def _resolve_override_weight_format(override_dir: Path, mode: str) -> tuple[bool | None, str, tuple[Path, ...], tuple[Path, ...]]:
    """Infer use_safetensors and layout for a component override directory.

    Sharded PyTorch exports use ``*.bin.index.json`` plus shard ``*.bin`` files; they must
    not be mistaken for "pick safetensors because some other file exists" and need
    ``use_safetensors=False`` so loaders follow the index.

    Returns:
        (use_safetensors, layout_tag, pytorch_bin_index_paths, safetensors_index_paths)
    """
    pytorch_bin_indices = tuple(sorted(override_dir.glob("*.bin.index.json")))
    safetensors_indices = tuple(sorted(override_dir.glob("*.safetensors.index.json")))

    if mode == "safetensors":
        if pytorch_bin_indices:
            print(
                f"Error: override directory has PyTorch sharded weights ({pytorch_bin_indices[0].name}) "
                "but --override-weight-format safetensors was requested.",
                file=sys.stderr,
            )
            sys.exit(1)
        return True, "safetensors_forced", pytorch_bin_indices, safetensors_indices

    if mode == "pytorch":
        if safetensors_indices and not pytorch_bin_indices:
            print(
                f"Warning: override directory has safetensors index ({safetensors_indices[0].name}) "
                "while --override-weight-format pytorch was requested; load may fail.",
                file=sys.stderr,
            )
        return False, "pytorch_forced", pytorch_bin_indices, safetensors_indices

    # auto
    if pytorch_bin_indices:
        return (
            False,
            f"pytorch_sharded[{pytorch_bin_indices[0].name}]",
            pytorch_bin_indices,
            safetensors_indices,
        )
    if safetensors_indices:
        return (
            True,
            f"safetensors_sharded[{safetensors_indices[0].name}]",
            pytorch_bin_indices,
            safetensors_indices,
        )

    has_st = any(override_dir.glob("*.safetensors"))
    has_bin = any(override_dir.glob("*.bin"))
    if has_st:
        return True, "safetensors", pytorch_bin_indices, safetensors_indices
    if has_bin:
        return False, "pytorch_bin", pytorch_bin_indices, safetensors_indices
    return None, "unspecified", pytorch_bin_indices, safetensors_indices


def _filter_pretrained_kwargs(cls, kw: dict) -> dict:
    """Drop unsupported kwargs for ``cls.from_pretrained`` (varies by diffusers/transformers)."""
    try:
        sig = inspect.signature(cls.from_pretrained)
    except (TypeError, ValueError):
        return kw
    params = sig.parameters
    return {k: v for k, v in kw.items() if k in params}


def replace_submodule(
    pipe,
    attr: str,
    path: Path,
    trust_remote_code: bool,
    label: str,
    torch_dtype=None,
    use_safetensors=None,
    layout_tag: str = "",
    pytorch_bin_index_paths: tuple[Path, ...] = (),
) -> None:
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

    def _load_with(kw: dict):
        filtered = _filter_pretrained_kwargs(cls, kw)
        return cls.from_pretrained(str(path), **filtered)

    def _is_sharded_pytorch_layout() -> bool:
        return bool(pytorch_bin_index_paths) or "pytorch_sharded" in layout_tag

    def _missing_monolithic_bin_message(msg: str) -> bool:
        lower = msg.lower()
        return (
            "diffusion_pytorch_model.bin" in lower
            or "pytorch_model.bin" in lower
        ) and ("not found" in lower or "no file named" in lower or "does not exist" in lower)

    try:
        replacement = _load_with(load_kw)
    except OSError as exc:
        if _is_sharded_pytorch_layout() and _missing_monolithic_bin_message(str(exc)):
            retry_kw = {**load_kw, "use_safetensors": False, "low_cpu_mem_usage": True}
            try:
                replacement = _load_with(retry_kw)
            except OSError as exc2:
                idx = pytorch_bin_index_paths[0] if pytorch_bin_index_paths else "(*.bin.index.json)"
                print(
                    f"Error: failed to load sharded PyTorch override for {label} from {path}\n"
                    f"  Index: {idx}\n"
                    f"  First error: {exc}\n"
                    f"  Retry error: {exc2}",
                    file=sys.stderr,
                )
                sys.exit(1)
        else:
            if pytorch_bin_index_paths:
                print(
                    f"Error: override load failed for {label} ({path}); "
                    f"directory contains PyTorch shard index {pytorch_bin_index_paths[0].name}: {exc}",
                    file=sys.stderr,
                )
            raise
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
    check_env()
    parser = argparse.ArgumentParser(
        description="Load a local diffusers pipeline (no download; use msdl first)"
    )
    parser.add_argument(
        "model",
        nargs="?",
        default=None,
        help=(
            "Pipeline name or path (relative to /workspace/models or absolute). "
            "If omitted, lists all pipelines found in /workspace/models."
        ),
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
        help="Weight format for override directories: auto detects from files "
             "(including *.bin.index.json / *.safetensors.index.json sharded layouts), "
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

    # No model specified: scan MODELS_ROOT and list available pipelines
    if args.model is None:
        if not MODELS_ROOT.is_dir():
            print(f"Error: models root does not exist: {MODELS_ROOT}", file=sys.stderr)
            sys.exit(1)
        found = sorted(
            d.name for d in MODELS_ROOT.iterdir()
            if d.is_dir() and (d / "model_index.json").is_file()
        )
        for name in found:
            print(f"[PIPELINE] {name}")
        sys.exit(0)

    base = resolve_model_path(args.model)
    torch_dtype = _parse_torch_dtype(args.torch_dtype)

    print(f"Model dir: {base}")
    print(f"Models root: {MODELS_ROOT}")
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
        use_st, layout, bin_idx_paths, _st_idx_paths = _resolve_override_weight_format(
            op, args.override_weight_format
        )
        print(
            f"Loading override {label} from {op} "
            f"(use_safetensors={use_st}, layout={layout}) ..."
        )
        replace_submodule(
            pipe,
            attr,
            op,
            args.trust_remote_code,
            label,
            torch_dtype=torch_dtype,
            use_safetensors=use_st,
            layout_tag=layout,
            pytorch_bin_index_paths=bin_idx_paths,
        )
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
