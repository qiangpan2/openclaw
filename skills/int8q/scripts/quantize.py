"""Quantize a local model to INT8 using llmcompressor (W8A8 or W8A16)."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def build_recipe(scheme: str, smoothing: float):
    from llmcompressor.modifiers.quantization import GPTQModifier
    from llmcompressor.modifiers.smoothquant import SmoothQuantModifier

    modifiers = []
    if scheme == "W8A8":
        modifiers.append(SmoothQuantModifier(smoothing_strength=smoothing))
    modifiers.append(
        GPTQModifier(targets="Linear", scheme=scheme, ignore=["lm_head"])
    )
    return modifiers


def _detect_split(dataset_id: str) -> str:
    from datasets import get_dataset_split_names

    splits = get_dataset_split_names(dataset_id, trust_remote_code=True)
    for candidate in ("train_sft", "train"):
        if candidate in splits:
            return candidate
    return splits[0]


def prepare_dataset(dataset_id: str, tokenizer, num_samples: int, max_seq_len: int):
    from datasets import load_dataset

    split = _detect_split(dataset_id)
    print(f"  Using split: {split}")
    ds = load_dataset(dataset_id, split=split, trust_remote_code=True)
    ds = ds.shuffle(seed=42).select(range(min(num_samples, len(ds))))

    columns = ds.column_names
    if "messages" in columns:
        def preprocess(example):
            return {
                "text": tokenizer.apply_chat_template(
                    example["messages"], tokenize=False
                )
            }
        ds = ds.map(preprocess)
    elif "text" in columns:
        pass
    else:
        print(
            f"Error: dataset has no 'messages' or 'text' column. "
            f"Found: {columns}",
            file=sys.stderr,
        )
        sys.exit(1)

    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=max_seq_len,
            padding=False,
        )

    return ds.map(tokenize, remove_columns=ds.column_names)


def main():
    parser = argparse.ArgumentParser(
        description="Quantize a local model to INT8 using llmcompressor"
    )
    parser.add_argument(
        "model", help="Local model path, e.g. /workspace/models/Qwen/Qwen3.5-0.8B"
    )
    parser.add_argument(
        "--scheme",
        choices=["W8A8", "W8A16"],
        default="W8A8",
        help="Quantization scheme (default: W8A8)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory (default: <model>-<scheme>)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=512,
        help="Number of calibration samples (default: 512)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=2048,
        help="Max calibration sequence length (default: 2048)",
    )
    parser.add_argument(
        "--dataset",
        default="HuggingFaceH4/ultrachat_200k",
        help="Calibration dataset (default: HuggingFaceH4/ultrachat_200k)",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.8,
        help="SmoothQuant smoothing strength, W8A8 only (default: 0.8)",
    )
    args = parser.parse_args()

    model_path = Path(args.model).resolve()
    if not model_path.is_dir():
        print(f"Error: model path does not exist: {model_path}", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output) if args.output else model_path.parent / f"{model_path.name}-{args.scheme}"
    output_path = output_path.resolve()

    print(f"Model:   {model_path}")
    print(f"Scheme:  {args.scheme}")
    print(f"Output:  {output_path}")
    print(f"Samples: {args.samples}  Seq-len: {args.seq_len}")
    print(f"Dataset: {args.dataset}")
    if args.scheme == "W8A8":
        print(f"Smoothing: {args.smoothing}")
    print()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)

    print("Loading model ...")
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), device_map="auto", torch_dtype="auto", trust_remote_code=True
    )

    print(f"Preparing calibration dataset ({args.samples} samples) ...")
    dataset = prepare_dataset(args.dataset, tokenizer, args.samples, args.seq_len)

    print("Building quantization recipe ...")
    recipe = build_recipe(args.scheme, args.smoothing)

    from llmcompressor import oneshot

    print("Running oneshot quantization (this may take a while) ...")
    oneshot(model=model, dataset=dataset, recipe=recipe)

    print(f"Saving quantized model to {output_path} ...")
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    report = {
        "source_model": str(model_path),
        "scheme": args.scheme,
        "calibration_samples": args.samples,
        "calibration_seq_len": args.seq_len,
        "calibration_dataset": args.dataset,
        "output_path": str(output_path),
        "quantized_at": datetime.now(timezone.utc).isoformat(),
    }
    if args.scheme == "W8A8":
        report["smoothing_strength"] = args.smoothing

    report_file = output_path / "quantize-report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Report written to {report_file}")

    print(f"\nDone. Quantized model saved to: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
