"""GPU-side data containers using NVIDIA Warp.

This module defines Warp struct containers for particle and gas data
that mirror the CPU-side ParticleData and GasData containers. These
structs enable efficient GPU-resident data for multi-box CFD simulations.

The structs use Warp array types for GPU memory management:
- wp.array: 1D arrays
- wp.array2d: 2D arrays
- wp.array3d: 3D arrays

All arrays support batch dimensions (n_boxes) for multi-box simulations.
Single-box simulations use n_boxes=1.

References:
    NVIDIA Warp documentation: https://nvidia.github.io/warp/
    Warp structs: https://nvidia.github.io/warp/modules/structs.html
"""

import warp as wp


@wp.struct
class WarpParticleData:
    """GPU-side particle data container using Warp arrays.

    All arrays have batch dimension for multi-box CFD support.
    Mirrors the shape convention of ParticleData from particula.particles.

    Batch Dimension Convention:
        - First dimension is always n_boxes (number of simulation boxes)
        - Single-box simulations use n_boxes=1
        - This enables vectorized operations across multiple boxes

    Attributes:
        masses: Per-species masses in kg.
            Shape: (n_boxes, n_particles, n_species)
            3D array where each particle has mass per species.
        concentration: Number concentration per particle.
            Shape: (n_boxes, n_particles)
            For particle-resolved: actual count (typically 1).
            For binned: number per m^3.
        charge: Particle charges (dimensionless integer counts).
            Shape: (n_boxes, n_particles)
            Stored as float64 for GPU compatibility.
        density: Material densities in kg/m^3.
            Shape: (n_species,)
            Shared across all boxes - not batched.
        volume: Simulation volume per box in m^3.
            Shape: (n_boxes,)

    Example:
        >>> import warp as wp
        >>> from particula.gpu import WarpParticleData
        >>> wp.init()
        >>> n_boxes, n_particles, n_species = 2, 100, 3
        >>> data = WarpParticleData()
        >>> data.masses = wp.zeros(
        ...     (n_boxes, n_particles, n_species), dtype=wp.float64
        ... )
        >>> data.concentration = wp.ones(
        ...     (n_boxes, n_particles), dtype=wp.float64
        ... )
        >>> data.charge = wp.zeros(
        ...     (n_boxes, n_particles), dtype=wp.float64
        ... )
        >>> data.density = wp.array(
        ...     [1000.0, 1200.0, 1500.0], dtype=wp.float64
        ... )
        >>> data.volume = wp.array([1e-3, 1e-3], dtype=wp.float64)
    """

    masses: wp.array3d(dtype=wp.float64)  # type: ignore[valid-type]
    concentration: wp.array2d(dtype=wp.float64)  # type: ignore[valid-type]
    charge: wp.array2d(dtype=wp.float64)  # type: ignore[valid-type]
    density: wp.array(dtype=wp.float64)  # type: ignore[valid-type]
    volume: wp.array(dtype=wp.float64)  # type: ignore[valid-type]


@wp.struct
class WarpGasData:
    """GPU-side gas species data container using Warp arrays.

    All batched arrays have shape (n_boxes, n_species) for multi-box CFD.
    Mirrors the GasData container from particula.gas, but excludes
    string fields (names) which are not GPU-compatible.

    Batch Dimension Convention:
        - First dimension is n_boxes (number of simulation boxes)
        - Second dimension is n_species (number of gas species)
        - Single-box simulations use n_boxes=1

    Note:
        - The 'name' field from CPU GasData is excluded (strings not
          GPU-compatible). Use index mapping for species identification.
        - The 'partitioning' field uses int32 instead of bool for GPU
          compatibility (1 = True, 0 = False).
        - 'vapor_pressure' is added for GPU kernels (not in CPU GasData).

    Attributes:
        molar_mass: Molar masses in kg/mol.
            Shape: (n_species,)
            Shared across all boxes - not batched.
        concentration: Mass concentrations in kg/m^3.
            Shape: (n_boxes, n_species)
        vapor_pressure: Vapor pressures in Pa.
            Shape: (n_boxes, n_species)
            Added for GPU condensation kernels.
        partitioning: Whether each species can partition to particles.
            Shape: (n_species,)
            Uses int32 (1=True, 0=False) for GPU compatibility.

    Example:
        >>> import warp as wp
        >>> from particula.gpu import WarpGasData
        >>> wp.init()
        >>> n_boxes, n_species = 2, 3
        >>> gas = WarpGasData()
        >>> gas.molar_mass = wp.array(
        ...     [0.018, 0.150, 0.200], dtype=wp.float64
        ... )
        >>> gas.concentration = wp.zeros(
        ...     (n_boxes, n_species), dtype=wp.float64
        ... )
        >>> gas.vapor_pressure = wp.ones(
        ...     (n_boxes, n_species), dtype=wp.float64
        ... ) * 1000.0
        >>> gas.partitioning = wp.array([1, 1, 0], dtype=wp.int32)
    """

    molar_mass: wp.array(dtype=wp.float64)  # type: ignore[valid-type]
    concentration: wp.array2d(dtype=wp.float64)  # type: ignore[valid-type]
    vapor_pressure: wp.array2d(dtype=wp.float64)  # type: ignore[valid-type]
    partitioning: wp.array(dtype=wp.int32)  # type: ignore[valid-type]
