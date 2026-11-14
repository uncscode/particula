# mypy: disable-error-code="operator"
"""Device ion transport approximation; see dynamics.ions.mobility_attachment."""

import warp as wp

from particula.gpu.dynamics.coagulation_funcs import (
    coulomb_continuum_limit_wp,
    coulomb_kinetic_limit_wp,
    coulomb_potential_ratio_wp,
    reduced_value_wp,
)
from particula.gpu.properties.gas_properties import (
    dynamic_viscosity_wp,
    molecule_mean_free_path_wp,
)
from particula.gpu.properties.particle_properties import (
    cunningham_slip_correction_wp,
)
from particula.util.constants import (
    BOLTZMANN_CONSTANT,
    ELECTRIC_PERMITTIVITY,
    ELEMENTARY_CHARGE_VALUE,
    GAS_CONSTANT,
    MOLECULAR_WEIGHT_AIR,
    REF_TEMPERATURE_STP,
    REF_VISCOSITY_AIR_STP,
    SUTHERLAND_CONSTANT,
)

_K = wp.constant(wp.float64(BOLTZMANN_CONSTANT))
_E = wp.constant(wp.float64(ELEMENTARY_CHARGE_VALUE))
_EPS = wp.constant(wp.float64(ELECTRIC_PERMITTIVITY))
_R = wp.constant(wp.float64(GAS_CONSTANT))
_MAIR = wp.constant(wp.float64(MOLECULAR_WEIGHT_AIR))
_MU = wp.constant(wp.float64(REF_VISCOSITY_AIR_STP))
_TREF = wp.constant(wp.float64(REF_TEMPERATURE_STP))
_S = wp.constant(wp.float64(SUTHERLAND_CONSTANT))
_PI = wp.constant(wp.float64(3.141592653589793))

_ION_TREF = wp.constant(wp.float64(298.15))
_FIT_A = wp.constant(wp.float64(25.836))
_FIT_B = wp.constant(wp.float64(11.211))
_FIT_C = wp.constant(wp.float64(3.502))
_FIT_D = wp.constant(wp.float64(7.211))
_CUTOFF = wp.constant(wp.float64(1e-80))


@wp.func
def mobility_attachment_wp(
    radius: wp.float64,
    mass: wp.float64,
    charge: wp.float64,
    temperature: wp.float64,
    pressure: wp.float64,
    ion_mass: wp.float64,
    ion_mobility: wp.float64,
    ion_sign: wp.float64,
    ion_radius: wp.float64,
) -> wp.float64:
    """Return SI attachment rate, or -1 for unsupported numeric inputs."""
    zero = wp.float64(0.0)
    if (
        not wp.isfinite(radius)
        or radius <= zero
        or not wp.isfinite(mass)
        or mass <= zero
        or not wp.isfinite(charge)
        or charge != wp.round(charge)
        or not wp.isfinite(temperature)
        or temperature <= zero
        or not wp.isfinite(pressure)
        or pressure <= zero
        or not wp.isfinite(ion_mass)
        or ion_mass <= zero
        or not wp.isfinite(ion_mobility)
        or ion_mobility <= zero
        or not wp.isfinite(ion_radius)
        or ion_radius <= zero
        or wp.abs(ion_sign) != wp.float64(1.0)
    ):
        return wp.float64(-1.0)
    mu = dynamic_viscosity_wp(temperature, _MU, _TREF, _S)
    path = molecule_mean_free_path_wp(_MAIR, temperature, pressure, mu, _R)
    slip = cunningham_slip_correction_wp(path / radius)
    f = wp.float64(6.0) * _PI * mu * radius / slip
    zi = (
        ion_mobility
        * wp.sqrt(temperature / _ION_TREF)
        * wp.float64(101325.0)
        / pressure
    )
    fi = _E / zi
    fr = reduced_value_wp(f, fi)
    mr = reduced_value_wp(mass, ion_mass)
    a = radius + ion_radius
    phi = coulomb_potential_ratio_wp(
        radius, ion_radius, charge, ion_sign, temperature, _K, _E, _EPS
    )
    kinetic = coulomb_kinetic_limit_wp(phi)
    if kinetic < _CUTOFF:
        return zero
    continuum = coulomb_continuum_limit_wp(phi)
    kn = wp.sqrt(_K * temperature * mr) / (fr * a) * continuum / kinetic
    # H / Kn^2 avoids the fourth-power intermediate of the dimensional fit.
    ratio = (
        wp.float64(4.0) * _PI
        + _FIT_A * kn
        + wp.sqrt(wp.float64(8.0) * _PI) * _FIT_B * kn * kn
    ) / (
        wp.float64(1.0) + _FIT_C * kn + _FIT_D * kn * kn + _FIT_B * kn * kn * kn
    )
    result = ratio * _K * temperature * a * continuum / fr
    if not wp.isfinite(result) or result < zero:
        return wp.float64(-1.0)
    return result
