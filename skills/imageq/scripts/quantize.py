"""Quantize a local diffusers submodule with torchao (W8A8 or W4A16)."""

import argparse
import gc
import importlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from paths import resolve_model_path


def parse_torch_dtype(raw: str):
    import torch

    mapping = {
        "auto": None,
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[raw]


def build_quantization_config(scheme: str, group_size: int):
    from diffusers import TorchAoConfig

    if scheme == "W8A8":
        return TorchAoConfig("int8dq"), "int8dq"
    if scheme == "W4A16":
        from torchao.quantization import Int4WeightOnlyConfig

        return TorchAoConfig(Int4WeightOnlyConfig(group_size=group_size)), "int4wo"

    raise ValueError(f"Unsupported scheme: {scheme}")


def _resolve_class_from_model_index(base: Path, target: str):
    """Parse model_index.json to resolve the concrete class for a pipeline slot."""
    index_path = base / "model_index.json"
    if not index_path.is_file():
        return None, None

    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    entry = index.get(target)
    if entry is None or not isinstance(entry, list) or len(entry) < 2:
        return None, None

    library_name, class_name = entry[0], entry[1]
    try:
        mod = importlib.import_module(library_name)
        cls = getattr(mod, class_name)
        return cls, class_name
    except (ImportError, AttributeError) as exc:
        print(
            f"Warning: model_index.json lists {library_name}.{class_name} for `{target}` "
            f"but import failed ({exc}); falling back to full pipeline load.",
            file=sys.stderr,
        )
        return None, None


def _discover_component_via_pipeline(base: Path, target: str, trust_remote_code: bool):
    """Fallback: load the full pipeline to discover a component's concrete class."""
    from diffusers import DiffusionPipeline

    print("Warning: falling back to full pipeline load to discover component type (high memory) ...",
          file=sys.stderr)
    pipe = DiffusionPipeline.from_pretrained(
        str(base), local_files_only=True, trust_remote_code=trust_remote_code,
    )
    component = getattr(pipe, target, None)
    if component is None:
        print(
            f"Error: this pipeline has no quantizable `{target}` component.",
            file=sys.stderr,
        )
        sys.exit(1)

    cls = type(component)
    cls_name = cls.__name__
    del component
    del pipe
    gc.collect()
    return cls, cls_name


def discover_component(base: Path, target: str, trust_remote_code: bool):
    """Resolve the concrete class for a pipeline slot, preferring model_index.json."""
    cls, cls_name = _resolve_class_from_model_index(base, target)
    if cls is not None:
        print(f"Resolved `{target}` as {cls_name} from model_index.json.")
        return cls, cls_name
    return _discover_component_via_pipeline(base, target, trust_remote_code)


def main():
    parser = argparse.ArgumentParser(
        description="Quantize a local diffusers submodule with torchao"
    )
    parser.add_argument(
        "model",
        help="Pipeline directory: relative to MODELSCOPE_CACHE or absolute",
    )
    parser.add_argument(
        "--scheme",
        choices=["W8A8", "W4A16"],
        default="W8A8",
        help="Quantization scheme: W8A8 (default, torchao int8dq) or W4A16 (torchao int4wo)",
    )
    parser.add_argument(
        "--target",
        choices=["transformer", "unet", "vae"],
        default="transformer",
        help="Pipeline component to quantize (default: transformer)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory for the quantized component (default: <model>-<target>-<scheme>)",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=128,
        help="TorchAO group size for W4A16 / int4wo (default: 128)",
    )
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="bfloat16",
        help="torch_dtype to pass during load (default: bfloat16)",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass trust_remote_code to from_pretrained (default: true)",
    )
    parser.add_argument(
        "--report",
        default=None,
        metavar="PATH",
        help="Write JSON report to this path (default: <output>/quantize-report.json)",
    )
    args = parser.parse_args()

    base = resolve_model_path(args.model)
    component_path = base / args.target
    if not component_path.is_dir():
        print(
            f"Error: expected component directory does not exist: {component_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    output_path = (
        Path(args.output).resolve()
        if args.output
        else (base.parent / f"{base.name}-{args.target}-{args.scheme}").resolve()
    )
    report_path = (
        Path(args.report).resolve()
        if args.report
        else output_path / "quantize-report.json"
    )
    torch_dtype = parse_torch_dtype(args.torch_dtype)
    quant_config, torchao_scheme = build_quantization_config(args.scheme, args.group_size)

    print(f"Pipeline dir: {base}")
    print(f"Component:    {args.target}")
    print(f"Scheme:       {args.scheme} -> torchao {torchao_scheme}")
    print(f"Source dir:   {component_path}")
    print(f"Output dir:   {output_path}")
    print(f"torch_dtype:  {args.torch_dtype}")
    if args.scheme == "W4A16":
        print(f"Group size:   {args.group_size}")
    print()

    cls, cls_name = discover_component(base, args.target, args.trust_remote_code)

    print(f"Reloading `{args.target}` as {cls_name} with torchao quantization ...")
    load_kw = {
        "local_files_only": True,
        "trust_remote_code": args.trust_remote_code,
        "quantization_config": quant_config,
    }
    if torch_dtype is not None:
        load_kw["torch_dtype"] = torch_dtype
    quantized = cls.from_pretrained(str(component_path), **load_kw)
    print("Quantized load OK.")

    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving quantized component to {output_path} ...")
    quantized.save_pretrained(str(output_path), safe_serialization=False)
    print("Save OK.")

    report = {
        "pipeline_path": str(base),
        "component_path": str(component_path),
        "target": args.target,
        "component_class": cls_name,
        "scheme": args.scheme,
        "torchao_scheme": torchao_scheme,
        "backend": "torchao",
        "group_size": args.group_size if args.scheme == "W4A16" else None,
        "torch_dtype": args.torch_dtype,
        "trust_remote_code": args.trust_remote_code,
        "output_path": str(output_path),
        "safe_serialization": False,
        "quantized_at": datetime.now(timezone.utc).isoformat(),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Report written to {report_path}")

    print(f"\nDone. Quantized component saved to: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
