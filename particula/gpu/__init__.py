"""GPU acceleration module for particula.

This module provides GPU-accelerated data containers and CPU↔Warp
conversion helpers using NVIDIA Warp for high-performance particle
simulations.

The GPU module enables:
- GPU-resident particle, gas, and environment data via Warp structs
- Efficient multi-box CFD simulations with batch dimensions
- Manual transfer control between CPU and GPU (to_warp/from_warp)
- Top-level availability and transfer helpers, while direct GPU step imports
  live under ``particula.gpu.kernels``

Example:
    >>> import warp as wp
    >>> from particula.gpu import WarpParticleData, WarpGasData
    >>> wp.init()
    >>> # Create particle data for 2 boxes, 100 particles, 3 species
    >>> data = WarpParticleData()
    >>> data.masses = wp.zeros((2, 100, 3), dtype=wp.float64)

References:
    NVIDIA Warp documentation: https://nvidia.github.io/warp/
"""

# pylint: disable=unused-import
# flake8: noqa
# pyright: basic

from __future__ import annotations

from importlib import import_module
from typing import Any


_WARP_IMPORT_ERROR: Exception | None = None


def _warp_unavailable_error() -> RuntimeError:
    """Build a helpful deferred error for Warp-backed symbol access."""
    message = (
        "Warp-backed GPU support is unavailable because importing 'warp' "
        "failed during particula.gpu initialization."
    )
    if _WARP_IMPORT_ERROR is None:
        return RuntimeError(message)
    return RuntimeError(f"{message} Original error: {_WARP_IMPORT_ERROR!r}")


def _check_warp_available() -> bool:
    """Check if Warp is available for GPU operations.

    Returns:
        True if warp can be imported, False otherwise.
    """
    global _WARP_IMPORT_ERROR

    try:
        import warp as wp  # noqa: F401

        _WARP_IMPORT_ERROR = None
        return True
    except Exception as exc:
        _WARP_IMPORT_ERROR = exc
        return False


WARP_AVAILABLE = _check_warp_available()

from particula.gpu.conversion import (
    from_warp_environment_data,
    from_warp_gas_data,
    from_warp_particle_data,
    gpu_context,
    to_warp_environment_data,
    to_warp_gas_data,
    to_warp_particle_data,
)

__all__ = [
    "WARP_AVAILABLE",
    "to_warp_particle_data",
    "to_warp_gas_data",
    "to_warp_environment_data",
    "from_warp_particle_data",
    "from_warp_gas_data",
    "from_warp_environment_data",
    "gpu_context",
]

if WARP_AVAILABLE:
    __all__.extend(
        [
            "WarpParticleData",
            "WarpGasData",
            "WarpEnvironmentData",
        ]
    )


def __getattr__(name: str) -> Any:
    """Lazily resolve optional Warp-backed container types."""
    warp_type_names = {
        "WarpParticleData",
        "WarpGasData",
        "WarpEnvironmentData",
    }
    if name not in warp_type_names:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if not WARP_AVAILABLE:
        raise _warp_unavailable_error() from _WARP_IMPORT_ERROR

    warp_types = import_module("particula.gpu.warp_types")
    value = getattr(warp_types, name)
    globals()[name] = value
    return value
