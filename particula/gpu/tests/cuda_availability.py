"""CUDA availability helpers shared across Warp-backed tests.

The helpers centralize the exported CUDA skip reason, suppress Warp's known
Python 3.14 ctypes warning during CUDA probing, and preserve the stable
``["cpu"]`` / ``["cpu", "cuda"]`` device enumeration contract used by the
GPU test suite.
"""

from __future__ import annotations

import warnings
from typing import Any

CUDA_SKIP_REASON = "Warp/CUDA not available"


def cuda_available(wp: Any) -> bool:
    """Return whether Warp can access CUDA with targeted warning suppression."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Due to '_pack_'.*",
            category=DeprecationWarning,
        )
        try:
            return bool(wp.is_cuda_available())
        except Exception:
            return False


def warp_devices(wp: Any) -> list[str]:
    """Return the stable Warp test device list.

    Args:
        wp: Warp module or Warp-like object exposing ``is_cuda_available``.

    Returns:
        ``["cpu"]`` when CUDA is unavailable, otherwise ``["cpu", "cuda"]``.
    """
    return ["cpu"] + (["cuda"] if cuda_available(wp) else [])
