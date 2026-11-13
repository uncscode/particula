"""Private CPU batch staging shared by native runnable implementations."""

from copy import copy
from numbers import Integral, Real

import numpy as np

from particula.aerosol import Aerosol
from particula.aerosol_validation import validate_aerosol_structure
from particula.gas import GasData
from particula.particles import ParticleData


def duration(time_step, sub_steps):
    """Validate scalar controls before allocating or evaluating state."""
    if (
        isinstance(sub_steps, (bool, np.bool_))
        or not isinstance(sub_steps, Integral)
        or sub_steps <= 0
    ):
        raise ValueError("sub_steps must be a positive integer")
    value = np.asarray(time_step)
    if value.ndim != 0:
        raise ValueError("time_step must be a finite nonnegative scalar")
    if isinstance(time_step, (bool, np.bool_)) or value.dtype.kind == "b":
        raise TypeError("time_step must be numeric, not boolean")
    if not isinstance(time_step, Real) and value.dtype.kind not in "iuf":
        raise TypeError("time_step must be numeric and real")
    if not np.isfinite(value) or value < 0:
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
        if target.size == 0:
            continue
        # A stride-span proof handles contiguous, transposed, reversed and
        # ordinary sliced arrays without a quadratic scan over box pairs.
        span = target.itemsize
        for stride, length in sorted(
            (abs(stride), length)
            for stride, length in zip(target.strides, target.shape, strict=True)
            if length > 1
        ):
            if stride < span:
                # Interleaved views can fail the simple span proof without
                # overlapping. Resolve only these unusual layouts exactly.
                offsets = np.array([0], dtype=np.int64)
                for size, step in zip(
                    target.shape, target.strides, strict=True
                ):
                    offsets = (
                        offsets[:, None] + np.arange(size) * step
                    ).ravel()
                offsets.sort()
                if np.any(np.diff(offsets) < target.itemsize):
                    raise ValueError("CPU writer elements must not overlap")
                break
            span += stride * (length - 1)


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
    # EnvironmentData's public constructor deliberately copies. A private
    # carrier must instead write through to this detached candidate only.
    environment = copy(e)
    environment.temperature = e.temperature[row]
    environment.pressure = e.pressure[row]
    environment.saturation_ratio = e.saturation_ratio[row]
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
        environment=environment,
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
