"""GPU acceleration module for particula.

This module provides GPU-accelerated data containers and operations
using NVIDIA Warp for high-performance particle simulations.

The GPU module enables:
- GPU-resident particle and gas data via Warp structs
- Efficient multi-box CFD simulations with batch dimensions
- Manual transfer control between CPU and GPU (to_warp/from_warp)
- GPU kernels for condensation (particle-resolved mass transfer)

Available GPU kernels:
- condensation_mass_transfer_kernel: Computes dm/dt for each particle
- apply_mass_transfer_kernel: Applies mass changes with clamping
- condensation_step_gpu: High-level function for one condensation timestep

Example:
    >>> import warp as wp
    >>> from particula.gpu import (
    ...     to_warp_particle_data,
    ...     to_warp_gas_data,
    ...     condensation_step_gpu,
    ... )
    >>> wp.init()
    >>> # Transfer data to GPU and run condensation
    >>> gpu_particles = to_warp_particle_data(particles, device="cuda")
    >>> gpu_gas = to_warp_gas_data(gas, device="cuda")
    >>> for _ in range(1000):
    ...     gpu_particles = condensation_step_gpu(
    ...         gpu_particles, gpu_gas,
    ...         temperature=298.15, dt=0.001
    ...     )

References:
    NVIDIA Warp documentation: https://nvidia.github.io/warp/
    Seinfeld & Pandis (2016): Atmospheric Chemistry and Physics
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

# Lazy import for optional dependency handling
if WARP_AVAILABLE:
    from particula.gpu.warp_types import WarpGasData, WarpParticleData
    from particula.gpu.conversion import (
        to_warp_particle_data,
        to_warp_gas_data,
        from_warp_particle_data,
        from_warp_gas_data,
        gpu_context,
    )
    from particula.gpu.kernels import (
        condensation_mass_transfer_kernel,
        apply_mass_transfer_kernel,
        condensation_step_gpu,
    )

__all__ = [
    "WARP_AVAILABLE",
    "WarpParticleData",
    "WarpGasData",
    "to_warp_particle_data",
    "to_warp_gas_data",
    "from_warp_particle_data",
    "from_warp_gas_data",
    "gpu_context",
    # GPU kernels
    "condensation_mass_transfer_kernel",
    "apply_mass_transfer_kernel",
    "condensation_step_gpu",
]
