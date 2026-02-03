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
    from particula.gpu.warp_types import WarpGasData, WarpParticleData
    from particula.particles.particle_data import ParticleData


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


def from_warp_particle_data(
    gpu_data: "WarpParticleData",
    sync: bool = True,
) -> "ParticleData":
    """Transfer WarpParticleData back to CPU.

    Use this to transfer GPU-resident particle data back to CPU after
    GPU simulation steps. The returned ParticleData can be used for
    checkpointing, analysis, or continuing with CPU-based operations.

    Args:
        gpu_data: GPU-resident WarpParticleData container.
        sync: If True (default), synchronize device before transfer
            to ensure all GPU operations have completed. Set False
            only if you've already synchronized manually (e.g., for
            batched transfers).

    Returns:
        CPU-side ParticleData with NumPy arrays.

    Example:
        >>> # After GPU simulation
        >>> result = from_warp_particle_data(gpu_data)
        >>>
        >>> # Batched transfers with manual sync (advanced)
        >>> import warp as wp
        >>> wp.synchronize()
        >>> data1 = from_warp_particle_data(gpu_data1, sync=False)
        >>> data2 = from_warp_particle_data(gpu_data2, sync=False)
    """
    wp = _ensure_warp_available()

    if sync:
        wp.synchronize()

    from particula.particles.particle_data import ParticleData

    return ParticleData(
        masses=gpu_data.masses.numpy(),
        concentration=gpu_data.concentration.numpy(),
        charge=gpu_data.charge.numpy(),
        density=gpu_data.density.numpy(),
        volume=gpu_data.volume.numpy(),
    )


def from_warp_gas_data(
    gpu_data: "WarpGasData",
    name: list | None = None,
    sync: bool = True,
) -> "GasData":
    """Transfer WarpGasData back to CPU.

    Use this to transfer GPU-resident gas data back to CPU after
    GPU simulation steps. The returned GasData can be used for
    checkpointing, analysis, or continuing with CPU-based operations.

    Note:
        The 'vapor_pressure' field from WarpGasData is not transferred
        as GasData does not have this field. If you need vapor pressure
        values, access them directly from gpu_data before calling this.

        Species names must be provided since WarpGasData does not store
        string data. If not provided, placeholder names are generated.

    Args:
        gpu_data: GPU-resident WarpGasData container.
        name: Species names. If None, generates placeholder names
            ["species_0", "species_1", ...].
        sync: If True (default), synchronize device before transfer
            to ensure all GPU operations have completed. Set False
            only if you've already synchronized manually.

    Returns:
        CPU-side GasData with NumPy arrays.

    Raises:
        ValueError: If name length doesn't match n_species.

    Example:
        >>> # With original names preserved
        >>> result = from_warp_gas_data(gpu_data, name=["Water", "H2SO4"])
        >>>
        >>> # With placeholder names
        >>> result = from_warp_gas_data(gpu_data)
        >>> print(result.name)  # ["species_0", "species_1"]
    """
    wp = _ensure_warp_available()

    if sync:
        wp.synchronize()

    # Determine n_species from molar_mass shape
    n_species = gpu_data.molar_mass.shape[0]

    # Handle name parameter
    if name is None:
        name = [f"species_{i}" for i in range(n_species)]
    elif len(name) != n_species:
        raise ValueError(
            f"name length {len(name)} does not match n_species {n_species}"
        )

    # Convert partitioning from int32 to bool
    partitioning_bool = gpu_data.partitioning.numpy().astype(bool)

    from particula.gas.gas_data import GasData

    return GasData(
        name=name,
        molar_mass=gpu_data.molar_mass.numpy(),
        concentration=gpu_data.concentration.numpy(),
        partitioning=partitioning_bool,
    )


class gpu_context:
    """Context manager for scoped GPU-resident simulation.

    Transfers ParticleData to GPU on entry. User is responsible for
    calling from_warp_particle_data() when ready to transfer back
    (typically inside the context or after exit).

    This is a convenience wrapper for simple GPU simulation patterns.
    For complex workflows with multiple data containers or explicit
    sync control, use to_warp_particle_data()/from_warp_particle_data()
    directly.

    Args:
        data: CPU-side ParticleData to transfer to GPU.
        device: Target GPU device ("cuda", "cuda:0", "cpu").

    Yields:
        WarpParticleData on the specified device.

    Example:
        >>> # Transfer back inside context
        >>> with gpu_context(particles) as gpu_data:
        ...     for _ in range(1000):
        ...         gpu_data = physics_step(gpu_data, dt)
        ...     result = from_warp_particle_data(gpu_data)
        >>>
        >>> # Or transfer back after context (data still valid)
        >>> with gpu_context(particles) as gpu_data:
        ...     for _ in range(1000):
        ...         gpu_data = physics_step(gpu_data, dt)
        >>> result = from_warp_particle_data(gpu_data)
    """

    def __init__(self, data: "ParticleData", device: str = "cuda"):
        """Initialize gpu_context.

        Args:
            data: CPU-side ParticleData to transfer to GPU.
            device: Target GPU device ("cuda", "cuda:0", "cpu").
        """
        self.data = data
        self.device = device
        self.gpu_data = None

    def __enter__(self) -> "WarpParticleData":
        """Transfer data to GPU and return GPU-resident data."""
        self.gpu_data = to_warp_particle_data(self.data, device=self.device)
        return self.gpu_data

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context. No automatic transfer back."""
        return False
