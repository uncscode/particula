"""Tests for Warp CUDA availability helpers."""

from __future__ import annotations

import warnings

from particula.gpu.tests.cuda_availability import cuda_available, warp_devices


class _WarningWarp:
    """Minimal Warp-like object that warns during CUDA checks."""

    def __init__(self, available: bool) -> None:
        self.available = available

    def is_cuda_available(self) -> bool:
        """Return availability while matching Warp's Python 3.14 warning."""
        warnings.warn(
            "Due to '_pack_', the 'APICLaunchParamRecord' Structure will use "
            "memory layout compatible with MSVC (Windows).",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.available


def test_cuda_available_ignores_warp_pack_warning() -> None:
    """CUDA helper suppresses the known Warp ctypes warning."""
    assert cuda_available(_WarningWarp(True))


def test_warp_devices_uses_cuda_availability() -> None:
    """Device list includes CUDA only when Warp reports it available."""
    assert warp_devices(_WarningWarp(True)) == ["cpu", "cuda"]
    assert warp_devices(_WarningWarp(False)) == ["cpu"]
