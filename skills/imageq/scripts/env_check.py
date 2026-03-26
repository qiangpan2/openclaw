"""Pre-flight environment compatibility check for imageq skill.

Called at the top of quantize.py and run.py before any model loading.
Also runnable standalone: python env_check.py

Exit codes: 0 = compatible, 1 = incompatible (prints [ENV ERROR] to stderr).
Prints [ENV OK] to stdout on success.
"""

import sys
from pathlib import Path

_SKILL_DIR = Path(__file__).resolve().parent.parent  # scripts/ -> imageq/
_FIX_CMD = (
    f"uv pip install -p {_SKILL_DIR}/.venv -r {_SKILL_DIR}/requirements.txt --reinstall"
)


def _fail(reason: str) -> None:
    print(f"[ENV ERROR] {reason}", file=sys.stderr)
    print(f"  → Fix: {_FIX_CMD}", file=sys.stderr)
    sys.exit(1)


def _parse_cuda_build_major(torch_version: str) -> int | None:
    """Return CUDA major version from torch version string, e.g. '2.6.0+cu126' -> 12."""
    if "+" in torch_version:
        local = torch_version.split("+", 1)[1]
        if local.startswith("cu") and len(local) >= 4:
            try:
                return int(local[2:4])
            except ValueError:
                pass
    return None


def check_env() -> None:
    """Run all pre-flight checks. Exits with [ENV ERROR] on failure."""
    import importlib.metadata

    # Collect installed versions
    for pkg in ("torch", "torchao", "diffusers"):
        try:
            importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            _fail(f"{pkg} not installed — venv may be incomplete")

    import torch as _torch
    torch_version = _torch.__version__
    torchao_version = importlib.metadata.version("torchao")
    diffusers_version = importlib.metadata.version("diffusers")

    # Check 1: torch CUDA build tag must be cu12x, not cu13x
    build_major = _parse_cuda_build_major(torch_version)
    if build_major is not None and build_major >= 13:
        _fail(
            f"torch {torch_version} uses CUDA {build_major}.x build; "
            f"system driver supports CUDA 12.x only\n"
            f"  → Need torch with +cu12x build (e.g. +cu126)"
        )

    # Check 2: driver version vs build major (warn only — cuda may not init cleanly)
    try:
        import torch
        if torch.cuda.is_available() and build_major is not None:
            drv = torch.cuda.driver_version()  # e.g. 12090 for 12.9
            drv_major = int(str(drv)[:2])
            if build_major > drv_major:
                _fail(
                    f"torch build CUDA {build_major} > driver CUDA {drv_major} "
                    f"(driver version {drv})\n"
                    f"  → Install torch with +cu{drv_major}x build"
                )
    except Exception:
        pass  # non-fatal: cuda init errors handled by later checks

    # Check 3: float8 symbol (requires torchao C++ extensions to be loaded)
    try:
        from torchao.quantization import float8_dynamic_activation_float8_weight  # noqa: F401
    except ImportError as exc:
        _fail(
            f"torchao {torchao_version} missing float8_dynamic_activation_float8_weight: {exc}\n"
            f"  → Likely CUDA build mismatch causes torchao to skip C++ extensions"
        )

    # Check 4: Int4WeightOnlyConfig (required for W4A16)
    try:
        from torchao.quantization import Int4WeightOnlyConfig  # noqa: F401
    except ImportError as exc:
        _fail(f"torchao {torchao_version} missing Int4WeightOnlyConfig: {exc}")

    # Check 5: TorchAoConfig("int8dq") construction (diffusers x torchao interface)
    try:
        from diffusers import TorchAoConfig
        TorchAoConfig("int8dq")
    except Exception as exc:
        _fail(
            f"TorchAoConfig('int8dq') failed "
            f"(diffusers={diffusers_version} torchao={torchao_version}): {exc}"
        )

    print(
        f"[ENV OK] torch={torch_version} torchao={torchao_version} diffusers={diffusers_version}"
    )


if __name__ == "__main__":
    check_env()
