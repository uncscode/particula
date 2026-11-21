"""Neutral Hamaker enhancement from the pair geometry at any radius ratio."""

from functools import lru_cache

import numpy as np
from particula import particles
from particula.dynamics.coagulation._pair_potential import hamaker_energy
from particula.dynamics.coagulation.charged_dimensionless_kernel import (
    get_hard_sphere_kernel,
)
from particula.util.constants import BOLTZMANN_CONSTANT
from scipy.integrate import trapezoid
from scipy.special import roots_laguerre


@lru_cache(maxsize=8192)
def enhancements(ratio, thermal_attraction, order=48):
    """Integrate continuum resistance and thermal ballistic cross section."""
    # A uniform grid in log gap resolves the short contact layer at high ratio.
    x = -np.expm1(-np.linspace(0, 35, 32 * order + 1))
    u = thermal_attraction * hamaker_energy(x, 1 / (1 + ratio))
    resistance = trapezoid(np.exp(u), x) + (1 - x[-1]) * np.exp(u[-1]) / 2
    t, weight = roots_laguerre(order)
    # At energy t*kT, each separation restricts the impact parameter.
    # Minimize over separation, then average the flux over incident energies.
    limit = (t[:, None] - u[None, 1:]) / x[None, 1:] ** 2
    return 1 / resistance, weight @ limit.min(axis=1)


def neutral_geometry(diameter_nm, hamaker, temp, pressure, density, order=48):
    """Return the complete thermal K (cm³/s) on a log-spaced diameter grid."""
    r = np.asarray(diameter_nm) * 0.5e-9
    log_step = np.diff(np.log(r))
    np.testing.assert_allclose(log_step, log_step[0], rtol=1e-11)
    kbt = BOLTZMANN_CONSTANT * temp
    ec, ef = np.array(
        [
            enhancements(np.exp(i * log_step[0]), hamaker / kbt, order)
            for i in range(len(r))
        ]
    ).T
    separation = np.abs(np.subtract.outer(np.arange(len(r)), np.arange(len(r))))
    ec, ef = ec[separation], ef[separation]
    m = 4 * np.pi / 3 * density * r**3
    mu = 1 / (1 / m[:, None] + 1 / m)
    diff = particles.get_diffusion_coefficient_via_system_state(
        r, temp, pressure
    )
    drag = kbt / (diff[:, None] + diff)
    contact = r[:, None] + r
    kn = np.sqrt(kbt * mu) / (drag * contact)
    return (
        1e6
        * drag
        * contact**3
        / mu
        * (ef**2 / ec)
        * get_hard_sphere_kernel(kn * ec / ef)
    )
