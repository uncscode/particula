"""Private CPU batch staging shared by native runnable implementations."""

from numbers import Integral, Real

import numpy as np

from particula.aerosol import Aerosol
from particula.aerosol_validation import validate_aerosol_structure
from particula.gas import EnvironmentData, GasData
from particula.particles import ParticleData


def duration(time_step, sub_steps):
    """Validate scalar controls before allocating or evaluating state."""
    if (
        isinstance(sub_steps, (bool, np.bool_))
        or not isinstance(sub_steps, Integral)
        or sub_steps <= 0
    ):
        raise ValueError("sub_steps must be a positive integer")
    if (
        isinstance(time_step, (bool, np.bool_))
        or not isinstance(time_step, Real)
        or not np.isfinite(time_step)
        or time_step < 0
    ):
        raise ValueError("time_step must be finite and nonnegative")
    return float(time_step) / int(sub_steps)


def arrays(aerosol):
    """List stored numeric arrays in a stable ownership order."""
    p, g, e = aerosol.particles, aerosol.gas, aerosol.environment
    result = {
        "particles.masses": p.masses,
        "particles.concentration": p.concentration,
        "particles.charge": p.charge,
        "particles.volume": p.volume,
        "particles.density": p.density,
        "gas.concentration": g.concentration,
        "gas.molar_mass": g.molar_mass,
        "gas.partitioning": g.partitioning,
        "environment.temperature": e.temperature,
        "environment.pressure": e.pressure,
        "environment.saturation_ratio": e.saturation_ratio,
    }
    if p.radius_grid is not None:
        result["particles.radius_grid"] = p.radius_grid
    return result


def validate(aerosol, writers=()):
    """Validate original storage before detached copies can hide defects."""
    if not isinstance(aerosol, Aerosol):
        raise TypeError("CPU runnables require a native Aerosol")
    validate_aerosol_structure(
        aerosol.particles, aerosol.gas, aerosol.environment
    )
    aerosol.particles.validate_representation()
    aerosol.environment.__post_init__()
    fields = arrays(aerosol)
    for name in writers:
        target = fields[name]
        if not target.flags.writeable:
            raise ValueError(f"{name} must be writable")
        if any(
            np.shares_memory(target, other)
            for key, other in fields.items()
            if key != name
        ):
            raise ValueError("CPU writer fields must not alias state")
        # Reject overlapping lanes, including writable zero-stride views.
        for box in range(target.shape[0]):
            if any(
                np.shares_memory(target[box : box + 1], target[j : j + 1])
                for j in range(box)
            ):
                raise ValueError("CPU writer box rows must not alias")


def detached(aerosol):
    """Stage independent containers preserving population metadata."""
    return Aerosol(
        particles=aerosol.particles.copy(),
        gas=aerosol.gas.copy(),
        environment=aerosol.environment.copy(),
    )


def box_view(aerosol, box):
    """Expose one staged box through existing native container types."""
    p, g, e = aerosol.particles, aerosol.gas, aerosol.environment
    row = slice(box, box + 1)
    return Aerosol(
        particles=ParticleData(
            masses=p.masses[row],
            concentration=p.concentration[row],
            charge=p.charge[row],
            density=p.density,
            volume=p.volume[row],
            distribution_type=p.distribution_type,
            radius_grid=p.radius_grid,
        ),
        gas=GasData(
            name=g.name,
            molar_mass=g.molar_mass,
            concentration=g.concentration[row],
            partitioning=g.partitioning,
        ),
        environment=EnvironmentData(
            temperature=e.temperature[row],
            pressure=e.pressure[row],
            saturation_ratio=e.saturation_ratio[row],
        ),
    )


def publish(aerosol, candidate, writers):
    """Publish selected fields by identity with rollback on write failure."""
    original, staged = arrays(aerosol), arrays(candidate)
    snapshots = {name: original[name].copy() for name in writers}
    written = []
    try:
        for name in writers:
            written.append(name)
            np.copyto(original[name], staged[name])
    except Exception:
        for name in reversed(written):
            original[name][...] = snapshots[name]
        raise
