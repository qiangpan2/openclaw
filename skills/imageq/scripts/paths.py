"""Shared path utilities for imageq scripts."""

import os
import sys
from pathlib import Path


def resolve_model_path(raw: str) -> Path:
    """Resolve a model path relative to MODELSCOPE_CACHE, or as absolute."""
    p = Path(raw)
    if not p.is_absolute():
        cache = os.environ.get("MODELSCOPE_CACHE", "/workspace/models")
        p = Path(cache) / p
    p = p.resolve()
    if not p.is_dir():
        print(f"Error: model path does not exist or is not a directory: {p}", file=sys.stderr)
        sys.exit(1)
    return p
