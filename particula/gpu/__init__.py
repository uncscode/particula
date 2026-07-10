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


def _check_warp_available() -> bool:
    """Check if Warp is available for GPU operations.

    Returns:
        True if warp can be imported, False otherwise.
    """
    try:
        import warp as wp  # noqa: F401

        return True
    except ImportError:
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

# Lazy import for optional dependency handling.
if WARP_AVAILABLE:
    from particula.gpu.warp_types import (
        WarpEnvironmentData,
        WarpGasData,
        WarpParticleData,
    )

    __all__.extend(
        [
            "WarpParticleData",
            "WarpGasData",
            "WarpEnvironmentData",
        ]
    )
