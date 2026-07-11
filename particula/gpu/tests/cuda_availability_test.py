"""Tests for Warp CUDA availability helpers."""

from __future__ import annotations

import warnings

from particula.gpu.tests.cuda_availability import (
    CUDA_SKIP_REASON,
    cuda_available,
    warp_devices,
)


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


def test_cuda_available_is_warning_clean_for_targeted_pack_warning() -> None:
    """Targeted Warp warning remains suppressed under warning capture."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("error", DeprecationWarning)
        assert cuda_available(_WarningWarp(True)) is True

    assert caught == []


def test_warp_devices_returns_cpu_only_without_cuda() -> None:
    """Device list keeps Warp CPU when CUDA is unavailable."""
    assert warp_devices(_WarningWarp(False)) == ["cpu"]


def test_warp_devices_returns_cpu_and_cuda_when_available() -> None:
    """Device list appends CUDA when Warp reports it available."""
    assert warp_devices(_WarningWarp(True)) == ["cpu", "cuda"]


def test_cuda_skip_reason_matches_shared_contract() -> None:
    """Shared CUDA skip reason stays deterministic for downstream tests."""
    assert CUDA_SKIP_REASON == "Warp/CUDA not available"
