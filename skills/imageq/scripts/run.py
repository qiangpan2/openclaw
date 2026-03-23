"""Load a local diffusers pipeline from disk (ModelScope layout via msdl). No downloads."""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def resolve_model_path(raw: str) -> Path:
    p = Path(raw)
    if not p.is_absolute():
        cache = os.environ.get("MODELSCOPE_CACHE", "/workspace/models")
        p = Path(cache) / p
    p = p.resolve()
    if not p.is_dir():
        print(f"Error: model path does not exist or is not a directory: {p}", file=sys.stderr)
        sys.exit(1)
    return p


def replace_submodule(pipe, attr: str, path: Path, trust_remote_code: bool, label: str) -> None:
    """Load weights with the same concrete class as the pipeline's existing submodule."""
    sub = getattr(pipe, attr, None)
    if sub is None:
        print(
            f"Error: this pipeline has no {label} to override (missing or None attribute `{attr}`).",
            file=sys.stderr,
        )
        sys.exit(1)
    cls = type(sub)
    replacement = cls.from_pretrained(
        str(path),
        local_files_only=True,
        trust_remote_code=trust_remote_code,
    )
    setattr(pipe, attr, replacement)


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

    print(f"Model dir: {base}")
    print(f"MODELSCOPE_CACHE: {os.environ.get('MODELSCOPE_CACHE', '/workspace/models')}")
    print()

    from diffusers import DiffusionPipeline

    load_kw = {
        "local_files_only": True,
        "trust_remote_code": args.trust_remote_code,
    }
    print("Loading DiffusionPipeline ...")
    pipe = DiffusionPipeline.from_pretrained(str(base), **load_kw)
    print("Load OK.")

    overrides = {}

    if args.unet:
        up = resolve_model_path(args.unet)
        print(f"Loading override UNet from {up} ...")
        replace_submodule(pipe, "unet", up, args.trust_remote_code, "UNet")
        overrides["unet"] = str(up)
        print("UNet override OK.")
    if args.transformer:
        tp = resolve_model_path(args.transformer)
        print(f"Loading override transformer from {tp} ...")
        replace_submodule(pipe, "transformer", tp, args.trust_remote_code, "transformer")
        overrides["transformer"] = str(tp)
        print("Transformer override OK.")
    if args.vae:
        vp = resolve_model_path(args.vae)
        print(f"Loading override VAE from {vp} ...")
        replace_submodule(pipe, "vae", vp, args.trust_remote_code, "VAE")
        overrides["vae"] = str(vp)
        print("VAE override OK.")

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
            "overrides": overrides,
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
