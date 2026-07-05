"""Convert simulation data containers between CPU and Warp representations.

This module provides helpers to transfer CPU-side particle, gas, and
environment containers to Warp arrays and reconstruct their CPU-side
equivalents. The conversion functions enable manual control over device
selection, copy behavior, and synchronization for long GPU-resident
simulations.

Example:
    >>> from particula.gpu import (
    ...     to_warp_environment_data,
    ...     to_warp_gas_data,
    ...     to_warp_particle_data,
    ... )
    >>> gpu_particles = to_warp_particle_data(particles, device="cuda")
    >>> gpu_gas = to_warp_gas_data(gas, device="cuda")
    >>> gpu_environment = to_warp_environment_data(environment, device="cuda")
    >>> # Run GPU simulation loop
    >>> for _ in range(10000):
    ...     gpu_particles = condensation_step(
    ...         gpu_particles, gpu_gas, gpu_environment, dt
    ...     )
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Generator

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from particula.gas.environment_data import EnvironmentData
    from particula.gas.gas_data import GasData
    from particula.gpu.warp_types import (
        WarpEnvironmentData,
        WarpGasData,
        WarpParticleData,
    )
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


def _from_numpy_zero_copy(wp, values, dtype, device: str):
    """Call ``wp.from_numpy`` with an explicit zero-copy preference.

    Some Warp versions accept ``copy=False`` explicitly, while older versions
    reject the keyword and use their built-in default behavior. Prefer the
    explicit call when supported so regression tests can guard the intended
    branch, but fall back for compatibility with older Warp releases.
    """
    try:
        return wp.from_numpy(values, dtype=dtype, device=device, copy=False)
    except TypeError as exc:
        if "unexpected keyword argument 'copy'" not in str(exc):
            raise
        return wp.from_numpy(values, dtype=dtype, device=device)


def _restore_partitioning_bool(
    partitioning_values: NDArray[np.int32] | NDArray[np.float64],
) -> NDArray[np.bool_]:
    """Validate GPU partitioning flags before restoring CPU bool values.

    Args:
        partitioning_values: GPU-restored partitioning values.

    Returns:
        Partitioning values restored as a NumPy bool array.

    Raises:
        ValueError: If any value is not one of ``0`` or ``1``.
    """
    if not np.all(np.isin(partitioning_values, (0, 1))):
        raise ValueError(
            "invalid partitioning restore values: partitioning must contain "
            "only binary 0/1 values before restoring GasData"
        )
    return partitioning_values.astype(bool)


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
        gpu_data.masses = _from_numpy_zero_copy(
            wp, data.masses, dtype=wp.float64, device=device
        )
        gpu_data.concentration = _from_numpy_zero_copy(
            wp, data.concentration, dtype=wp.float64, device=device
        )
        gpu_data.charge = _from_numpy_zero_copy(
            wp, data.charge, dtype=wp.float64, device=device
        )
        gpu_data.density = _from_numpy_zero_copy(
            wp, data.density, dtype=wp.float64, device=device
        )
        gpu_data.volume = _from_numpy_zero_copy(
            wp, data.volume, dtype=wp.float64, device=device
        )

    return gpu_data


def to_warp_environment_data(
    data: "EnvironmentData",
    device: str = "cuda",
    copy: bool = True,
) -> "WarpEnvironmentData":
    """Transfer EnvironmentData to GPU with explicit control.

    Use this helper when environment state should remain on the selected
    Warp device alongside particle or gas data during GPU-resident
    simulations.

    Args:
        data: CPU-side EnvironmentData container.
        device: Target device ("cuda", "cuda:0", "cuda:1", "cpu").
        copy: If True (default), always copy data to device.
            If False, attempt zero-copy via wp.from_numpy() when
            arrays are already on a compatible device.

    Returns:
        WarpEnvironmentData with Warp arrays on specified device.

    Raises:
        RuntimeError: If Warp is not available or device not found.

    Example:
        >>> from particula.gpu import to_warp_environment_data
        >>> gpu_environment = to_warp_environment_data(
        ...     environment_data, device="cpu"
        ... )
        >>> gpu_environment.temperature.shape
        (1,)
    """
    wp = _ensure_warp_available()
    _validate_device(wp, device)

    from particula.gpu.warp_types import WarpEnvironmentData

    gpu_data = WarpEnvironmentData()

    if copy:
        gpu_data.temperature = wp.array(
            data.temperature, dtype=wp.float64, device=device
        )
        gpu_data.pressure = wp.array(
            data.pressure, dtype=wp.float64, device=device
        )
        gpu_data.saturation_ratio = wp.array(
            data.saturation_ratio, dtype=wp.float64, device=device
        )
    else:
        gpu_data.temperature = _from_numpy_zero_copy(
            wp, data.temperature, dtype=wp.float64, device=device
        )
        gpu_data.pressure = _from_numpy_zero_copy(
            wp, data.pressure, dtype=wp.float64, device=device
        )
        gpu_data.saturation_ratio = _from_numpy_zero_copy(
            wp, data.saturation_ratio, dtype=wp.float64, device=device
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
        gpu_data.molar_mass = _from_numpy_zero_copy(
            wp, data.molar_mass, dtype=wp.float64, device=device
        )
        gpu_data.concentration = _from_numpy_zero_copy(
            wp, data.concentration, dtype=wp.float64, device=device
        )
        gpu_data.vapor_pressure = _from_numpy_zero_copy(
            wp, vp_array, dtype=wp.float64, device=device
        )
        gpu_data.partitioning = _from_numpy_zero_copy(
            wp, partitioning_int, dtype=wp.int32, device=device
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
        The ``vapor_pressure`` field from ``WarpGasData`` is not restored
        because ``GasData`` does not include this field. If you need vapor
        pressure values, access them directly from ``gpu_data`` before
        calling this helper. ``from_warp_gas_data()`` restores only the
        CPU-owned ``GasData`` fields.

        ``WarpGasData`` does not store species names because string data is
        not GPU-compatible. Prefer caller-supplied ordered names when
        reconstructing ``GasData``. Placeholder names are generated only
        when ``name`` is omitted or explicitly set to ``None``.

        Restored ``partitioning`` values must remain binary ``0``/``1`` on
        the GPU side. Any non-binary values raise ``ValueError`` before a
        ``GasData`` instance is returned.

    Args:
        gpu_data: GPU-resident WarpGasData container.
        name: Optional ordered species names supplied by the caller.
            If omitted or ``None``, generates placeholder names such as
            ``["species_0", "species_1", ...]``. If provided, the list
            length must match ``gpu_data.molar_mass.shape[0]``.
        sync: If True (default), synchronize device before transfer
            to ensure all GPU operations have completed. Set False
            only if you've already synchronized manually.

    Returns:
        CPU-side GasData with NumPy arrays.

    Raises:
        ValueError: If the supplied name count does not match ``n_species``.
        ValueError: If restored ``partitioning`` contains values other than
            ``0`` or ``1``.

    Example:
        >>> # With caller-supplied ordered names
        >>> result = from_warp_gas_data(gpu_data, name=["Water", "H2SO4"])
        >>>
        >>> # With placeholder names because GPU data stores no names
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
            "name length mismatch when restoring GasData: expected "
            f"{n_species} names, got {len(name)}"
        )

    # Convert partitioning from int32 to bool after validating GPU flags.
    partitioning_bool = _restore_partitioning_bool(
        gpu_data.partitioning.numpy()
    )

    from particula.gas.gas_data import GasData

    # Intentionally drop GPU-only vapor_pressure when reconstructing GasData.
    return GasData(
        name=name,
        molar_mass=gpu_data.molar_mass.numpy(),
        concentration=gpu_data.concentration.numpy(),
        partitioning=partitioning_bool,
    )


def from_warp_environment_data(
    gpu_data: "WarpEnvironmentData",
    sync: bool = True,
) -> "EnvironmentData":
    """Transfer WarpEnvironmentData back to CPU.

    Use this to transfer GPU-resident environment data back to CPU after
    GPU simulation steps. The returned EnvironmentData can be used for
    checkpointing, analysis, or continuing with CPU-based operations.

    Args:
        gpu_data: GPU-resident WarpEnvironmentData container.
        sync: If True (default), synchronize device before transfer
            to ensure all GPU operations have completed. Set False
            only if you've already synchronized manually (e.g., for
            batched transfers).

    Returns:
        CPU-side EnvironmentData with NumPy arrays.

    Raises:
        ValueError: If the recovered arrays do not satisfy the
            ``EnvironmentData`` CPU schema.

    Example:
        >>> result = from_warp_environment_data(gpu_environment)
        >>>
        >>> import warp as wp
        >>> wp.synchronize()
        >>> result = from_warp_environment_data(gpu_environment, sync=False)
    """
    wp = _ensure_warp_available()

    if sync:
        wp.synchronize()

    from particula.gas.environment_data import EnvironmentData

    return EnvironmentData(
        temperature=gpu_data.temperature.numpy(),
        pressure=gpu_data.pressure.numpy(),
        saturation_ratio=gpu_data.saturation_ratio.numpy(),
    )


@contextmanager
def gpu_context(
    data: "ParticleData",
    device: str = "cuda",
) -> Generator["WarpParticleData", None, None]:
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
    gpu_data = to_warp_particle_data(data, device=device)
    yield gpu_data
