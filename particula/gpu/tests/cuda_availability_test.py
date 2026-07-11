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


class _UnexpectedWarningWarp:
    """Warp-like object that emits a non-targeted deprecation warning."""

    def is_cuda_available(self) -> bool:
        """Raise an unrelated warning that should remain visible."""
        warnings.warn(
            "unexpected warp deprecation",
            DeprecationWarning,
            stacklevel=2,
        )
        return True


class _ProbeErrorWarp:
    """Warp-like object whose CUDA probe raises unexpectedly."""

    def is_cuda_available(self) -> bool:
        """Simulate a Warp probe crash during collection-time probing."""
        raise RuntimeError("probe failed")


def test_cuda_available_ignores_warp_pack_warning() -> None:
    """CUDA helper suppresses the known Warp ctypes warning."""
    assert cuda_available(_WarningWarp(True))


def test_cuda_available_is_warning_clean_for_targeted_pack_warning() -> None:
    """Targeted Warp warning remains suppressed under warning capture."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("error", DeprecationWarning)
        assert cuda_available(_WarningWarp(True)) is True

    assert caught == []


def test_cuda_available_preserves_unrelated_deprecation_warnings() -> None:
    """Unexpected warning-to-error probe failures degrade to unavailable."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert cuda_available(_UnexpectedWarningWarp()) is False


def test_warp_devices_returns_cpu_only_without_cuda() -> None:
    """Device list keeps Warp CPU when CUDA is unavailable."""
    assert warp_devices(_WarningWarp(False)) == ["cpu"]


def test_warp_devices_returns_cpu_and_cuda_when_available() -> None:
    """Device list appends CUDA when Warp reports it available."""
    assert warp_devices(_WarningWarp(True)) == ["cpu", "cuda"]


def test_cuda_available_returns_false_when_probe_raises() -> None:
    """Unexpected CUDA probe errors degrade to the CPU-only path."""
    assert cuda_available(_ProbeErrorWarp()) is False


def test_warp_devices_remains_cpu_only_when_probe_raises() -> None:
    """Device enumeration should stay collection-safe when probing fails."""
    assert warp_devices(_ProbeErrorWarp()) == ["cpu"]


def test_cuda_skip_reason_matches_shared_contract() -> None:
    """Shared CUDA skip reason stays deterministic for downstream tests."""
    assert CUDA_SKIP_REASON == "Warp/CUDA not available"
