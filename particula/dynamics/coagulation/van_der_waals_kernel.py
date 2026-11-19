"""Thermal capture by nonretarded dispersion and polarizable spheres."""

from functools import lru_cache

import numpy as np
from scipy.integrate import trapezoid

from particula.dynamics.coagulation._pair_potential import (
    dielectric_response,
    hamaker_energy,
)
from particula.dynamics.coagulation.charged_dimensionless_kernel import (
    get_hard_sphere_kernel,
)
from particula.gas import get_dynamic_viscosity, get_molecule_mean_free_path
from particula.particles import (
    get_cunningham_slip_correction,
    get_friction_factor,
    get_knudsen_number,
)
from particula.util.constants import (
    BOLTZMANN_CONSTANT,
    ELEMENTARY_CHARGE_VALUE,
    VACUUM_PERMITTIVITY,
)

from .coagulation_strategy._native_validation import positive_scalar

# Lobatto endpoints resolve both the distant Coulomb tail and near contact.
RADIAL_INTERVALS = 128
RATIO_INTERVALS = 128
MAX_RADIUS_RATIO = 32.0
LOG_RATIO_STEP = np.log(MAX_RADIUS_RATIO) / RATIO_INTERVALS
DISTANCES = np.sin(np.linspace(0, np.pi / 2, RADIAL_INTERVALS + 1)) ** 2
COULOMB_LENGTH_T = ELEMENTARY_CHARGE_VALUE**2 / (
    4 * np.pi * VACUUM_PERMITTIVITY * BOLTZMANN_CONSTANT
)
TABLE_VERSION = 1


def validate_material(hamaker_constant, relative_permittivity):
    """Validate material scalars for nonretarded dielectric-sphere capture."""
    positive_scalar(hamaker_constant, "hamaker_constant", zero_allowed=True)
    positive_scalar(relative_permittivity, "relative_permittivity")
    if not 1 <= relative_permittivity <= 100:
        raise ValueError("relative_permittivity must lie in [1, 100]")


@lru_cache(maxsize=256)
def dielectric_row(relative_permittivity, index):
    """Tabulate a material's geometrical response at one radius ratio."""
    fraction = 1 / (1 + np.exp(index * LOG_RATIO_STEP))
    row = np.zeros((RADIAL_INTERVALS + 1, 3))
    if relative_permittivity != 1:
        for k, x in enumerate(DISTANCES[1:], 1):
            # The reflected field contains higher angular orders near contact.
            order = 16 if x < 0.8 else (40 if x < 0.97 else 96)
            row[k] = dielectric_response(
                x, fraction, relative_permittivity, order
            )
    row.flags.writeable = False
    return row


def _ballistic_enhancement(energy, barrier, distances):
    """Integrate the lower envelope of all angular momentum constraints.

    At kinetic energy t in thermal units, each radius imposes
    t*b²/R² <= (t-U/kT)/x². The active constraint moves outward in x
    order as t increases. Integrate this piecewise linear minimum against
    exp(-t) analytically, starting above the radial repulsion barrier.
    """
    nodes = []
    starts = []
    slopes = 1 / distances[1:] ** 2
    intercepts = -energy[1:] * slopes
    for i in range(len(slopes)):
        if not np.isfinite(intercepts[i]):
            continue
        while nodes:
            j = nodes[-1]
            start = (intercepts[i] - intercepts[j]) / (slopes[j] - slopes[i])
            if start > starts[-1]:
                break
            nodes.pop()
            starts.pop()
        nodes.append(i)
        starts.append(start if len(nodes) > 1 else -np.inf)
    low = np.maximum(starts, barrier)
    width = np.diff(low, append=np.inf)
    # Integration by parts avoids subtracting two nearly equal moments.
    at_barrier = max(0.0, np.min(barrier * slopes + intercepts))
    return np.exp(-barrier) * at_barrier + np.sum(
        slopes[nodes] * np.exp(-low) * -np.expm1(-width)
    )


def pair_enhancements(radius1, radius2, z1, z2, temperature, hamaker, epsilon):
    """Return continuum and free-molecular rate ratios for one pair."""
    if radius1 > radius2:
        radius1, radius2, z1, z2 = radius2, radius1, z2, z1
    contact = radius1 + radius2
    ratio = radius2 / radius1
    coupling = COULOMB_LENGTH_T / (temperature * contact)
    attraction = hamaker / (BOLTZMANN_CONSTANT * temperature)
    if (
        ratio > MAX_RADIUS_RATIO
        or attraction > 200
        or coupling * max(abs(z1), abs(z2)) ** 2 > 60
    ):
        raise ValueError("pair exceeds van der Waals capture table domain")
    bare = coupling * z1 * z2
    if attraction == 0 and (epsilon == 1 or (z1 == 0 and z2 == 0)):
        continuum = bare / np.expm1(bare) if bare != 0 else 1.0
        ballistic = np.exp(-bare) if bare > 0 else 1 - bare
        return continuum, ballistic

    distances = DISTANCES
    if attraction > 0:
        # Resolve the shrinking near-contact layer continuously as A -> 0.
        distances = DISTANCES.copy()
        gap = min((1 - DISTANCES[-2]) / 2, np.sqrt(attraction))
        distances[-1] = min(np.nextafter(1.0, 0.0), 1 - gap)
    energy = bare * distances
    if epsilon != 1 and (z1 != 0 or z2 != 0):
        coordinate = np.log(ratio) / LOG_RATIO_STEP
        index = min(int(coordinate), RATIO_INTERVALS - 1)
        weight = coordinate - index
        left = dielectric_row(epsilon, index)
        right = dielectric_row(epsilon, index + 1)
        induced = (
            coupling
            * ((1 - weight) * left + weight * right)
            @ [z1**2, z1 * z2, z2**2]
        )
        if attraction > 0:
            weight = (1 - distances[-1]) / (1 - DISTANCES[-2])
            induced[-1] += weight * (induced[-2] - induced[-1])
        energy += induced
    if attraction > 0:
        energy += attraction * hamaker_energy(distances, radius1 / contact)
    barrier = np.max(energy)
    boltzmann = np.exp(energy - barrier)
    resistance = (
        trapezoid(boltzmann, distances)
        + (1 - distances[-1]) * boltzmann[-1] / 2
    )
    return np.exp(-barrier) / resistance, _ballistic_enhancement(
        energy, barrier, distances
    )


def get_van_der_waals_kernel_via_system_state(
    particle_radius,
    particle_mass,
    particle_charge,
    temperature,
    pressure,
    hamaker_constant,
    relative_permittivity,
):
    """Return the symmetric thermal encounter matrix in m³/s.

    Args:
        particle_radius: Physical sphere radii [m], scalar or vector.
        particle_mass: Per-particle masses [kg], matching the radii.
        particle_charge: Signed numbers of elementary charges, matching radii.
        temperature: Gas and particle temperature [K].
        pressure: Gas pressure [Pa].
        hamaker_constant: Effective dispersion constant across air [J].
        relative_permittivity: Relative static permittivity of both particles.

    Returns:
        Square kernel array for compact polarizable spheres with unit sticking.

    References:
        Ouyang et al. (2012), doi:10.1063/1.4742064.
        Bichoutskaia et al. (2010), doi:10.1063/1.3457157.
    """
    validate_material(hamaker_constant, relative_permittivity)
    positive_scalar(temperature, "temperature")
    positive_scalar(pressure, "pressure")
    inputs = [
        np.atleast_1d(a)
        for a in (particle_radius, particle_mass, particle_charge)
    ]
    if any(
        a.dtype.kind not in "fiu" or a.ndim != 1 or not np.all(np.isfinite(a))
        for a in inputs
    ):
        raise ValueError("radius, mass and charge must be finite real vectors")
    radius, mass, charge = (a.astype(np.float64, copy=False) for a in inputs)
    if mass.shape != radius.shape or charge.shape != radius.shape:
        raise ValueError("radius, mass and charge must have equal length")
    if np.any(radius <= 0) or np.any(mass <= 0):
        raise ValueError("radius and mass must be positive")
    # Repeated resolved material states have identical pair rates. Evaluate
    # the distinct states once, then restore the requested slot layout.
    states, slot = np.unique(
        np.column_stack((radius, mass, charge)), axis=0, return_inverse=True
    )
    radius, mass, charge = states.T
    viscosity = get_dynamic_viscosity(temperature)
    path = get_molecule_mean_free_path(
        temperature=temperature, pressure=pressure, dynamic_viscosity=viscosity
    )
    slip = get_cunningham_slip_correction(get_knudsen_number(path, radius))
    friction = get_friction_factor(radius, viscosity, slip)
    result = np.empty((len(radius), len(radius)))
    for i in range(len(radius)):
        for j in range(i + 1):
            contact = radius[i] + radius[j]
            if (COULOMB_LENGTH_T / (contact * temperature)) * charge[
                i
            ] * charge[j] < -0.5:
                raise ValueError(
                    "opposite-charge attraction exceeds the transition domain"
                )
            ec, ef = pair_enhancements(
                radius[i],
                radius[j],
                charge[i],
                charge[j],
                temperature,
                hamaker_constant,
                relative_permittivity,
            )
            mu = mass[i] / (1 + mass[i] / mass[j])
            drag = friction[i] / (1 + friction[i] / friction[j])
            kn = np.sqrt(BOLTZMANN_CONSTANT * temperature * mu) / (
                drag * contact
            )
            h = get_hard_sphere_kernel(kn * ec / ef)
            result[i, j] = result[j, i] = (
                drag * contact**3 / mu * (ef**2 / ec) * h
            )
    return result[np.ix_(slot, slot)]
