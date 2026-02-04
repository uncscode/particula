"""GPU kernels for particle physics simulations.

This module provides GPU-accelerated kernels for condensation,
coagulation, and other particle dynamics using NVIDIA Warp.

The kernels use 2D launch patterns for multi-box CFD support:
- dim=(n_boxes, n_particles) for particle-level operations
- Each thread handles one (box, particle) pair

Available kernels:
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
    >>> # Run condensation on GPU
    >>> gpu_particles = condensation_step_gpu(
    ...     particles, gas, temperature=298.15, pressure=101325.0, dt=0.001
    ... )

References:
    NVIDIA Warp kernels: https://nvidia.github.io/warp/modules/kernels.html
    Seinfeld & Pandis (2016): Atmospheric Chemistry and Physics
"""

from particula.gpu.kernels.condensation import (
    apply_mass_transfer_kernel,
    condensation_mass_transfer_kernel,
    condensation_step_gpu,
)

__all__ = [
    "condensation_mass_transfer_kernel",
    "apply_mass_transfer_kernel",
    "condensation_step_gpu",
]
