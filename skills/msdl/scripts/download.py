# /// script
# requires-python = ">=3.10"
# dependencies = ["modelscope"]
# ///
"""Download a model from ModelScope to a local directory."""

import argparse
import os
import sys


def main():
    default_cache = os.environ.get("MODELSCOPE_CACHE", "/workspace/models")
    os.environ.setdefault("MODELSCOPE_CACHE", default_cache)

    parser = argparse.ArgumentParser(description="Download a model from ModelScope")
    parser.add_argument("model", help="Model ID, e.g. stepfun-ai/Step-3.5-Flash")
    parser.add_argument(
        "--local-dir",
        default=None,
        help="Download directly to this directory instead of the cache",
    )
    parser.add_argument(
        "--include",
        nargs="*",
        default=None,
        help="Glob patterns for files to include (e.g. '*.safetensors')",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=None,
        help="Glob patterns for files to exclude (e.g. '*.bin')",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Model revision (branch, tag, or commit hash)",
    )
    args = parser.parse_args()

    from modelscope import snapshot_download  # noqa: E402

    kwargs = {"model_id": args.model}
    if args.local_dir:
        kwargs["local_dir"] = args.local_dir
    else:
        kwargs["cache_dir"] = default_cache
    if args.revision:
        kwargs["revision"] = args.revision
    if args.include:
        kwargs["allow_patterns"] = args.include
    if args.exclude:
        kwargs["ignore_patterns"] = args.exclude

    print(f"Downloading {args.model} ...")
    model_path = snapshot_download(**kwargs)
    print(f"Done. Model saved to: {model_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
