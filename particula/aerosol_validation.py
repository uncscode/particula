"""Read-only structural alignment of CPU particle, gas, and environment data.

Import this concrete boundary directly; it does not admit a physical process.
"""

import numpy as np

from particula.gas.environment_data import EnvironmentData
from particula.gas.gas_data import GasData
from particula.particles.particle_data import ParticleData


def _require_shape(value: object, field: str, shape: tuple[int, ...]) -> None:
    """Require a raw NumPy array with the indicated rank and exact shape.

    Args:
        value: Stored field to inspect without conversion.
        field: Name to include in validation errors.
        shape: Required array dimensions.

    Raises:
        ValueError: If the field is not an array of the required shape.
    """
    if not isinstance(value, np.ndarray) or value.ndim != len(shape):
        raise ValueError(f"{field} must be an ndarray of shape {shape}")
    if value.shape != shape:
        raise ValueError(f"{field} shape must be {shape}; got {value.shape}")


def _gas_species_count(names: object) -> int:
    """Count the nonempty, unique gas names without changing their order.

    Args:
        names: Stored gas-name metadata to inspect.

    Returns:
        Number of ordered gas species.

    Raises:
        ValueError: If names are missing, blank, or duplicated.
    """
    if not isinstance(names, list) or not names:
        raise ValueError("gas.name must be a nonempty list")
    seen: set[str] = set()
    for name in names:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("gas.name entries must be nonblank strings")
        if name in seen:
            raise ValueError("gas.name entries must be unique")
        seen.add(name)
    return len(names)


def _require_containers(
    particles: object, gas: object, environment: object
) -> None:
    """Check all top-level types before inspecting any stored fields.

    Args:
        particles: Candidate particle container.
        gas: Candidate gas container.
        environment: Candidate environment container.

    Raises:
        TypeError: If an input is not its required CPU container.
    """
    if not isinstance(particles, ParticleData):
        raise TypeError("particles must be ParticleData")
    if not isinstance(gas, GasData):
        raise TypeError("gas must be GasData")
    if not isinstance(environment, EnvironmentData):
        raise TypeError("environment must be EnvironmentData")


def validate_aerosol_structure(
    particles: ParticleData,
    gas: GasData,
    environment: EnvironmentData,
) -> None:
    """Check raw CPU container dimensions and full ordered gas-name metadata.

    This check does not inspect array values, partitioning eligibility,
    distribution interpretation, or chemical provenance. In particular, equal
    ratio and gas widths do not prove that their species order agrees.

    Args:
        particles: Particle storage with nonempty box axis.
        gas: Ordered gas storage, including nonpartitioning species.
        environment: Per-box state with ratio lanes for every gas species.

    Returns:
        None when the stored schemas and shared dimensions agree.

    Raises:
        TypeError: If any top-level input is not its required CPU container.
        ValueError: If stored fields or cross-container shapes are malformed.
    """
    _require_containers(particles, gas, environment)

    masses = particles.masses
    if not isinstance(masses, np.ndarray) or masses.ndim != 3:
        raise ValueError("particles.masses must be a rank-3 ndarray")
    boxes, slots, particle_species = masses.shape
    if boxes == 0:
        raise ValueError("particles.masses box count must be positive")
    _require_shape(
        particles.concentration, "particles.concentration", (boxes, slots)
    )
    _require_shape(particles.charge, "particles.charge", (boxes, slots))
    _require_shape(particles.density, "particles.density", (particle_species,))
    _require_shape(particles.volume, "particles.volume", (boxes,))

    gas_species = _gas_species_count(gas.name)
    _require_shape(gas.molar_mass, "gas.molar_mass", (gas_species,))
    if (
        not isinstance(gas.concentration, np.ndarray)
        or gas.concentration.ndim != 2
    ):
        raise ValueError("gas.concentration must be a rank-2 ndarray")
    gas_boxes, concentration_species = gas.concentration.shape
    if concentration_species != gas_species:
        raise ValueError("gas.concentration species width must match gas.name")
    _require_shape(gas.partitioning, "gas.partitioning", (gas_species,))

    if (
        not isinstance(environment.temperature, np.ndarray)
        or environment.temperature.ndim != 1
    ):
        raise ValueError("environment.temperature must be a rank-1 ndarray")
    environment_boxes = environment.temperature.shape[0]
    _require_shape(
        environment.pressure, "environment.pressure", (environment_boxes,)
    )
    _require_shape(
        environment.saturation_ratio,
        "environment.saturation_ratio",
        (environment_boxes, gas_species),
    )
    if gas_boxes != boxes:
        raise ValueError(
            "gas.concentration box count must match particles.masses"
        )
    if environment_boxes != boxes:
        raise ValueError(
            "environment.temperature box count must match particles.masses"
        )
