"""Shared path utilities for imageq scripts."""

import sys
from pathlib import Path

MODELS_ROOT = Path("/workspace/models")


def resolve_model_path(raw: str) -> Path:
    """Resolve a model path relative to MODELS_ROOT, or as absolute."""
    p = Path(raw)
    if not p.is_absolute():
        p = MODELS_ROOT / p
    p = p.resolve()
    if not p.is_dir():
        print(f"Error: model path does not exist or is not a directory: {p}", file=sys.stderr)
        sys.exit(1)
    return p
