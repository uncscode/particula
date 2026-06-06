"""CUDA availability helpers for Warp tests."""

from __future__ import annotations

import warnings
from typing import Any


def cuda_available(wp: Any) -> bool:
    """Return whether Warp can access CUDA without failing on warnings."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Due to '_pack_'.*",
            category=DeprecationWarning,
        )
        return bool(wp.is_cuda_available())


def warp_devices(wp: Any) -> list[str]:
    """Return CPU plus CUDA when CUDA is available to Warp."""
    return ["cpu"] + (["cuda"] if cuda_available(wp) else [])
