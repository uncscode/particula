"""Coulomb transport approximation with independently specified ion mobility.

Low-field, constant collision-cross-section scaling gives Z proportional to
sqrt(T)/P (Mason--Schamp with ideal-gas number density). This is not LF2013:
image forces, hydration and three-body trapping are omitted. The transition
fit is equation (5) of Gopalakrishnan and Hogan (2012),
https://doi.org/10.1103/PhysRevE.85.026410. For the mobility relation see the
primary interlaboratory study https://doi.org/10.1021/acs.analchem.7b01729.
"""

import numpy as np

from particula import gas, particles
from particula.dynamics.coagulation.charged_dimensionless_kernel import (
    get_dimensional_kernel,
    get_hard_sphere_kernel,
)
from particula.util.constants import (
    BOLTZMANN_CONSTANT,
    ELECTRIC_PERMITTIVITY,
    ELEMENTARY_CHARGE_VALUE,
    GAS_CONSTANT,
    MOLECULAR_WEIGHT_AIR,
)


def mobility_attachment(
    radius,
    mass,
    charge,
    temperature,
    pressure,
    ion_mass,
    ion_mobility,
    ion_sign,
    ion_radius=0.25e-9,
):
    """Return broadcast ion-aerosol attachment coefficients [m^3/s].

    Args:
        radius: Aerosol volume-equivalent radius [m].
        mass: Aerosol total mass [kg].
        charge: Integer aerosol charge [elementary charges].
        temperature: Gas temperature [K].
        pressure: Air pressure [Pa].
        ion_mass: Singly charged ion mass [kg].
        ion_mobility: Mobility at 298.15 K, 101325 Pa [m^2/(V s)].
        ion_sign: +1 or -1.
        ion_radius: Ion contact radius [m], independent of mobility.

    Coulomb potential ratios are clipped to [-200, 200], and kinetic
    enhancement below 1e-80 is treated as zero, matching device guards.

    Returns:
        Broadcast nonnegative attachment coefficient array [m^3/s].

    Raises:
        ValueError: For invalid inputs or unrepresentable rates.
    """
    values = np.broadcast_arrays(
        radius,
        mass,
        charge,
        temperature,
        pressure,
        ion_mass,
        ion_mobility,
        ion_sign,
        ion_radius,
    )
    r, m, q, t, p, mi, z, sign, ri = values
    if (
        any(np.any(~np.isfinite(v)) for v in values)
        or any(np.any(v <= 0) for v in (r, m, t, p, mi, z, ri))
        or np.any(np.abs(sign) != 1)
        or np.any(q != np.round(q))
    ):
        raise ValueError("invalid mobility attachment inputs")
    viscosity = gas.get_dynamic_viscosity(t)
    mean_free_path = (2 * viscosity / p) / np.sqrt(
        8 * MOLECULAR_WEIGHT_AIR / (np.pi * GAS_CONSTANT * t)
    )
    slip = particles.get_cunningham_slip_correction(mean_free_path / r)
    friction = 6 * np.pi * viscosity * r / slip
    ion_friction = ELEMENTARY_CHARGE_VALUE / (
        z * np.sqrt(t / 298.15) * 101325 / p
    )
    reduced_friction = friction * ion_friction / (friction + ion_friction)
    reduced_mass = m * mi / (m + mi)
    radius_sum = r + ri
    phi = np.clip(
        -q
        * sign
        * ELEMENTARY_CHARGE_VALUE**2
        / (
            4
            * np.pi
            * ELECTRIC_PERMITTIVITY
            * radius_sum
            * BOLTZMANN_CONSTANT
            * t
        ),
        -200,
        200,
    )
    kinetic = particles.get_coulomb_kinetic_limit(phi)
    continuum = particles.get_coulomb_continuum_limit(phi)
    kn = (
        np.sqrt(BOLTZMANN_CONSTANT * t * reduced_mass)
        / (reduced_friction * radius_sum)
        * continuum
        / kinetic
    )
    rate = get_dimensional_kernel(
        get_hard_sphere_kernel(kn),
        phi,
        radius_sum,
        reduced_mass,
        reduced_friction,
    )
    rate = np.where(kinetic < 1e-80, 0, rate)
    if np.any(~np.isfinite(rate)) or np.any(rate < 0):
        raise ValueError("unrepresentable mobility attachment rate")
    return rate
