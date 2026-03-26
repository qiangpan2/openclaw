"""Quantize a local diffusers submodule with torchao (W8A8 or W4A16)."""

import argparse
import gc
import importlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import shutil

from paths import resolve_model_path, MODELS_ROOT
from env_check import check_env


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


def _assemble_pipeline_skeleton(src: Path, dst: Path, targets: list[str]) -> None:
    """Copy all non-target pipeline contents into dst.

    - Files at pipeline root: shutil.copy2
    - Dirs in targets: skipped (will be quantized separately)
    - All other dirs: shutil.copytree (full copy, no symlinks)
    """
    dst.mkdir(parents=True, exist_ok=True)

    for item in src.iterdir():
        dest_item = dst / item.name
        if item.is_file():
            shutil.copy2(item, dest_item)
        elif item.is_dir():
            if item.name in targets:
                continue  # will be quantized and saved separately
            shutil.copytree(item, dest_item)


def main():
    check_env()
    parser = argparse.ArgumentParser(
        description="Quantize a local diffusers pipeline with torchao (full pipeline output)"
    )
    parser.add_argument(
        "model",
        help="Pipeline directory: relative to /workspace/models or absolute",
    )
    parser.add_argument(
        "--scheme",
        choices=["W8A8", "W4A16"],
        default="W8A8",
        help="Quantization scheme: W8A8 (default, torchao int8dq) or W4A16 (torchao int4wo)",
    )
    parser.add_argument(
        "--target",
        nargs="+",
        default=["transformer", "vae"],
        metavar="COMPONENT",
        help="Pipeline components to quantize (default: transformer vae)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output pipeline directory "
            "(default: /workspace/models/<model-basename>-<scheme>)"
        ),
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

    # Validate all target component dirs exist before starting
    for target in args.target:
        component_path = base / target
        if not component_path.is_dir():
            print(
                f"Error: component directory does not exist: {component_path}",
                file=sys.stderr,
            )
            sys.exit(1)

    # Compute output path: MODELS_ROOT/<basename>-<scheme>
    output_path = (
        Path(args.output).resolve()
        if args.output
        else (MODELS_ROOT / f"{base.name}-{args.scheme}").resolve()
    )
    report_path = (
        Path(args.report).resolve()
        if args.report
        else output_path / "quantize-report.json"
    )

    torch_dtype = parse_torch_dtype(args.torch_dtype)
    quant_config, torchao_scheme = build_quantization_config(args.scheme, args.group_size)

    print(f"Source pipeline: {base}")
    print(f"Output pipeline: {output_path}")
    print(f"Scheme:          {args.scheme} -> torchao {torchao_scheme}")
    print(f"Targets:         {', '.join(args.target)}")
    print(f"torch_dtype:     {args.torch_dtype}")
    if args.scheme == "W4A16":
        print(f"Group size:      {args.group_size}")
    print()

    # Guard: refuse to overwrite an existing output directory
    if output_path.exists():
        print(
            f"Error: output directory already exists: {output_path}\n"
            "Remove it or choose a different --output path.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Step 1: copy all non-target pipeline contents
    print("Copying pipeline skeleton ...")
    _assemble_pipeline_skeleton(base, output_path, args.target)
    print()

    # Step 2: quantize each target component
    components_report: dict = {}
    for target in args.target:
        component_path = base / target
        cls, cls_name = discover_component(base, target, args.trust_remote_code)

        print(f"Quantizing `{target}` ({cls_name}) ...")
        load_kw: dict = {
            "local_files_only": True,
            "trust_remote_code": args.trust_remote_code,
            "quantization_config": quant_config,
        }
        if torch_dtype is not None:
            load_kw["torch_dtype"] = torch_dtype

        quantized = cls.from_pretrained(str(component_path), **load_kw)
        out_component = output_path / target
        print(f"Saving quantized `{target}` to {out_component} ...")
        quantized.save_pretrained(str(out_component), safe_serialization=False)
        del quantized
        gc.collect()

        components_report[target] = {
            "class": cls_name,
            "source": str(component_path),
        }
        print(f"`{target}` done.\n")

    # Step 3: write report at pipeline root
    report = {
        "source_pipeline": str(base),
        "output_pipeline": str(output_path),
        "scheme": args.scheme,
        "torchao_scheme": torchao_scheme,
        "torch_dtype": args.torch_dtype,
        "group_size": args.group_size if args.scheme == "W4A16" else None,
        "quantized_at": datetime.now(timezone.utc).isoformat(),
        "components": components_report,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Report written to {report_path}")
    print(f"\nDone. Quantized pipeline saved to: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
