"""Conversion functions between CPU and GPU data containers.

This module provides functions to transfer CPU-side ParticleData and GasData
containers to GPU using NVIDIA Warp arrays. The conversion functions enable
manual control over device selection and copy behavior for long GPU-resident
simulations.

Example:
    >>> from particula.gpu import to_warp_particle_data, to_warp_gas_data
    >>> gpu_particles = to_warp_particle_data(particles, device="cuda")
    >>> gpu_gas = to_warp_gas_data(gas, device="cuda")
    >>> # Run GPU simulation loop
    >>> for _ in range(10000):
    ...     gpu_particles = condensation_step(gpu_particles, gpu_gas, dt)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from particula.gas.gas_data import GasData
    from particula.particles.particle_data import ParticleData

    from particula.gpu.warp_types import WarpGasData, WarpParticleData


def _ensure_warp_available():
    """Ensure Warp is available, raise RuntimeError if not.

    Returns:
        The warp module if available.

    Raises:
        RuntimeError: If Warp is not installed.
    """
    try:
        import warp as wp

        return wp
    except ImportError as e:
        raise RuntimeError(
            "Warp is not installed. Install with: pip install warp-lang"
        ) from e


def _validate_device(wp, device: str):
    """Validate device string and return device object.

    Args:
        wp: The warp module.
        device: Device string (e.g., "cuda", "cuda:0", "cpu").

    Returns:
        Warp device object.

    Raises:
        RuntimeError: If device is not found.
    """
    try:
        return wp.get_device(device)
    except (RuntimeError, ValueError) as e:
        raise RuntimeError(
            f"Device '{device}' not found. Available devices: "
            "cuda, cuda:0, cuda:1, ..., cpu"
        ) from e


def to_warp_particle_data(
    data: "ParticleData",
    device: str = "cuda",
    copy: bool = True,
) -> "WarpParticleData":
    """Transfer ParticleData to GPU with explicit control.

    Use this for long GPU-resident simulations where you want to:
    1. Transfer data to GPU once at simulation start
    2. Run many timesteps without CPU round-trips
    3. Transfer back only when needed (checkpoints, final result)

    Args:
        data: CPU-side ParticleData container.
        device: Target device ("cuda", "cuda:0", "cuda:1", "cpu").
        copy: If True (default), always copy data to device.
              If False, attempt zero-copy via wp.from_numpy() when
              arrays are already on a compatible device.

    Returns:
        WarpParticleData with Warp arrays on specified device.

    Raises:
        RuntimeError: If Warp is not available or device not found.

    Example:
        >>> from particula.gpu import to_warp_particle_data
        >>> gpu_data = to_warp_particle_data(particles, device="cuda")
        >>> for _ in range(10000):
        ...     gpu_data = condensation_step(gpu_data, gas, dt)
        >>> result = from_warp_particle_data(gpu_data)
    """
    wp = _ensure_warp_available()
    _validate_device(wp, device)

    from particula.gpu.warp_types import WarpParticleData

    gpu_data = WarpParticleData()

    if copy:
        gpu_data.masses = wp.array(data.masses, dtype=wp.float64, device=device)
        gpu_data.concentration = wp.array(
            data.concentration, dtype=wp.float64, device=device
        )
        gpu_data.charge = wp.array(data.charge, dtype=wp.float64, device=device)
        gpu_data.density = wp.array(
            data.density, dtype=wp.float64, device=device
        )
        gpu_data.volume = wp.array(data.volume, dtype=wp.float64, device=device)
    else:
        # Zero-copy via wp.from_numpy() (if array is on compatible device)
        gpu_data.masses = wp.from_numpy(
            data.masses, dtype=wp.float64, device=device
        )
        gpu_data.concentration = wp.from_numpy(
            data.concentration, dtype=wp.float64, device=device
        )
        gpu_data.charge = wp.from_numpy(
            data.charge, dtype=wp.float64, device=device
        )
        gpu_data.density = wp.from_numpy(
            data.density, dtype=wp.float64, device=device
        )
        gpu_data.volume = wp.from_numpy(
            data.volume, dtype=wp.float64, device=device
        )

    return gpu_data


def to_warp_gas_data(
    data: "GasData",
    device: str = "cuda",
    copy: bool = True,
    vapor_pressure: NDArray[np.float64] | None = None,
) -> "WarpGasData":
    """Transfer GasData to GPU with explicit control.

    Use this for long GPU-resident simulations where you want to:
    1. Transfer data to GPU once at simulation start
    2. Run many timesteps without CPU round-trips
    3. Transfer back only when needed (checkpoints, final result)

    Note:
        The 'name' field from GasData is excluded (strings are not
        GPU-compatible). Use index mapping for species identification.

        The 'partitioning' field is converted from bool to int32
        (1 = True, 0 = False) for GPU compatibility.

        The 'vapor_pressure' field is required by WarpGasData but not
        present in GasData. If not provided, it defaults to zeros.
        Set it after conversion if needed for condensation kernels.

    Args:
        data: CPU-side GasData container.
        device: Target device ("cuda", "cuda:0", "cuda:1", "cpu").
        copy: If True (default), always copy data to device.
              If False, attempt zero-copy via wp.from_numpy() when
              arrays are already on a compatible device.
        vapor_pressure: Optional vapor pressure array in Pa.
            Shape: (n_boxes, n_species). If None, zeros are used.

    Returns:
        WarpGasData with Warp arrays on specified device.

    Raises:
        RuntimeError: If Warp is not available or device not found.
        ValueError: If vapor_pressure shape doesn't match expected
            (n_boxes, n_species).

    Example:
        >>> from particula.gpu import to_warp_gas_data
        >>> gpu_gas = to_warp_gas_data(gas_data, device="cuda")
        >>> # With explicit vapor pressure:
        >>> vp = np.array([[1000.0, 500.0, 200.0], [1000.0, 500.0, 200.0]])
        >>> gpu_gas = to_warp_gas_data(gas_data, vapor_pressure=vp)
    """
    wp = _ensure_warp_available()
    _validate_device(wp, device)

    from particula.gpu.warp_types import WarpGasData

    # Validate vapor_pressure shape if provided
    expected_shape = (data.n_boxes, data.n_species)
    if vapor_pressure is not None:
        if vapor_pressure.shape != expected_shape:
            raise ValueError(
                f"vapor_pressure shape {vapor_pressure.shape} does not match "
                f"expected {expected_shape}"
            )
        vp_array = vapor_pressure
    else:
        # Default to zeros
        vp_array = np.zeros(expected_shape, dtype=np.float64)

    # Convert partitioning from bool to int32 (1=True, 0=False)
    partitioning_int = data.partitioning.astype(np.int32)

    gpu_data = WarpGasData()

    if copy:
        gpu_data.molar_mass = wp.array(
            data.molar_mass, dtype=wp.float64, device=device
        )
        gpu_data.concentration = wp.array(
            data.concentration, dtype=wp.float64, device=device
        )
        gpu_data.vapor_pressure = wp.array(
            vp_array, dtype=wp.float64, device=device
        )
        gpu_data.partitioning = wp.array(
            partitioning_int, dtype=wp.int32, device=device
        )
    else:
        # Zero-copy via wp.from_numpy() (if array is on compatible device)
        gpu_data.molar_mass = wp.from_numpy(
            data.molar_mass, dtype=wp.float64, device=device
        )
        gpu_data.concentration = wp.from_numpy(
            data.concentration, dtype=wp.float64, device=device
        )
        gpu_data.vapor_pressure = wp.from_numpy(
            vp_array, dtype=wp.float64, device=device
        )
        gpu_data.partitioning = wp.from_numpy(
            partitioning_int, dtype=wp.int32, device=device
        )

    return gpu_data
