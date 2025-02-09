"""
Compute the square of the RMS fluctuation velocity and the cross-correlation
of the fluctuating velocities.
"""

from typing import Union, NamedTuple

import numpy as np
from numpy.typing import NDArray

from particula.util.validate_inputs import validate_inputs
from particula.dynamics.coagulation.turbulent_dns_kernel.velocity_correlation_terms_ao2008 import (
    compute_b1,
    compute_b2,
    compute_c1,
    compute_c2,
    compute_d1,
    compute_d2,
    compute_e1,
    compute_e2,
    compute_z,
    compute_beta,
)
from particula.dynamics.coagulation.turbulent_dns_kernel.psi_ao2008 import (
    get_psi_ao2008,
)
from particula.dynamics.coagulation.turbulent_dns_kernel.phi_ao2008 import (
    get_phi_ao2008,
)
from particula.dynamics.coagulation.turbulent_dns_kernel.velocity_correlation_f2_ao2008 import (
    get_f2_longitudinal_velocity_correlation,
)


class VelocityCorrelationTerms(NamedTuple):
    """Parameters from computing velocity correlation terms."""

    b1: Union[float, NDArray[np.float64]]
    b2: Union[float, NDArray[np.float64]]
    d1: Union[float, NDArray[np.float64]]
    d2: Union[float, NDArray[np.float64]]
    c1: Union[float, NDArray[np.float64]]
    c2: Union[float, NDArray[np.float64]]
    e1: Union[float, NDArray[np.float64]]
    e2: Union[float, NDArray[np.float64]]


@validate_inputs(
    {
        "fluid_rms_velocity": "positive",
        "collisional_radius": "positive",
        "particle_inertia_time": "positive",
        "particle_velocity": "positive",
    }
)
def get_relative_velocity_variance(
    fluid_rms_velocity: float,
    collisional_radius: NDArray[np.float64],
    particle_inertia_time: NDArray[np.float64],
    particle_velocity: NDArray[np.float64],
    taylor_microscale: float,
    eulerian_integral_length: float,
    lagrangian_integral_time: float,
    lagrangian_taylor_microscale_time: float,
) -> Union[float, NDArray[np.float64]]:
    # pylint: disable=too-many-arguments, disable=too-many-positional-arguments
    """
    Compute the variance of particle relative-velocity fluctuation.

    The function is given by:

        σ² = ⟨ (v'^(2))² ⟩ + ⟨ (v'^(1))² ⟩ - 2 ⟨ v'^(1) v'^(2) ⟩

    - ⟨ (v'^(k))² ⟩ is the square of the RMS fluctuation velocity for droplet k
    - ⟨ v'^(1) v'^(2) ⟩ is the cross-correlation of the fluctuating velocities.

    Arguments:
    ----------
        - fluid_rms_velocity : Fluid RMS fluctuation velocity [m/s].
        - collisional_radius : Distance between two colliding droplets [m].
        - particle_inertia_time : Inertia timescale of droplet 1 [s].
        - particle_velocity : Droplet velocity [m/s].
        - taylor_microscale : Taylor microscale [m].
        - eulerian_integral_length : Eulerian integral length scale [m].
        - lagrangian_integral_time : Lagrangian integral time scale [s].

    Returns:
    --------
        - σ² : Variance of particle relative-velocity fluctuation,
            (n, n) matrix where n is number of particles [m²/s²],

    References:
    -----------
    - Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on
        the geometric collision rate of sedimenting droplets. Part 2.
        Theory and parameterization. New Journal of Physics, 10.
        https://doi.org/10.1088/1367-2630/10/7/075016
    """

    z = compute_z(lagrangian_taylor_microscale_time, lagrangian_integral_time)
    beta = compute_beta(taylor_microscale, eulerian_integral_length)

    vel_corr_terms = VelocityCorrelationTerms(
        b1=compute_b1(z),
        b2=compute_b2(z),
        d1=compute_d1(beta),
        d2=compute_d2(beta),
        c1=compute_c1(z, lagrangian_integral_time),
        c2=compute_c2(z, lagrangian_integral_time),
        e1=compute_e1(z, eulerian_integral_length),
        e2=compute_e2(z, eulerian_integral_length),
    )

    rms_velocity = _compute_rms_fluctuation_velocity(
        fluid_rms_velocity, particle_inertia_time, vel_corr_terms
    )

    cross_correlation = _compute_cross_correlation_velocity(
        fluid_rms_velocity,
        collisional_radius,
        particle_inertia_time,
        particle_velocity,
        taylor_microscale,
        eulerian_integral_length,
        vel_corr_terms,
    )
    return rms_velocity[:, np.newaxis]**2 + rms_velocity[np.newaxis, :]**2 - 2 * cross_correlation


@validate_inputs(
    {
        "fluid_rms_velocity": "positive",
        "particle_inertia_time": "positive",
    }
)
def _compute_rms_fluctuation_velocity(
    fluid_rms_velocity: float,
    particle_inertia_time: Union[float, NDArray[np.float64]],
    velocity_correlation_terms: VelocityCorrelationTerms,
) -> Union[float, NDArray[np.float64]]:
    """
    Compute the square of the RMS fluctuation velocity for the k-th droplet.

    The function is given by:

        ⟨ (v'^(k))² ⟩ = (u'² / τ_pk) *\
                        [b₁ d₁ Ψ(c₁, e₁) - b₁ d₂ Ψ(c₁, e₂)\
                         - b₂ d₁ Ψ(c₂, e₁) + b₂ d₂ Ψ(c₂, e₂)]

    - u' (fluid_rms_velocity) : Fluid RMS fluctuation velocity [m/s].
    - τ_pk (particle_inertia_time) : Inertia timescale of the droplet k [s].
    - Ψ(c, e) : Function Ψ computed using `get_psi_ao2008`.

    Arguments:
    ----------
        - fluid_rms_velocity : Fluid RMS fluctuation velocity [m/s].
        - particle_inertia_time : Inertia timescale of the droplet k [s].
        - velocity_correlation_terms : Velocity correlation coefficients [-].

    Returns:
    --------
        - RMS fluctuation velocity squared ⟨ (v'^(k))² ⟩ [m²/s²].

    References:
    -----------
    - Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on
        the geometric collision rate of sedimenting droplets. Part 2.
        Theory and parameterization. New Journal of Physics, 10.
        https://doi.org/10.1088/1367-2630/10/7/075016
    """
    psi_c1_e1 = get_psi_ao2008(
        velocity_correlation_terms.c1,
        velocity_correlation_terms.e1,
        particle_inertia_time,
        fluid_rms_velocity,
    )
    psi_c1_e2 = get_psi_ao2008(
        velocity_correlation_terms.c1,
        velocity_correlation_terms.e2,
        particle_inertia_time,
        fluid_rms_velocity,
    )
    psi_c2_e1 = get_psi_ao2008(
        velocity_correlation_terms.c2,
        velocity_correlation_terms.e1,
        particle_inertia_time,
        fluid_rms_velocity,
    )
    psi_c2_e2 = get_psi_ao2008(
        velocity_correlation_terms.c2,
        velocity_correlation_terms.e2,
        particle_inertia_time,
        fluid_rms_velocity,
    )

    return (fluid_rms_velocity**2 / particle_inertia_time) * (
        velocity_correlation_terms.b1
        * velocity_correlation_terms.d1
        * psi_c1_e1
        - velocity_correlation_terms.b1
        * velocity_correlation_terms.d2
        * psi_c1_e2
        - velocity_correlation_terms.b2
        * velocity_correlation_terms.d1
        * psi_c2_e1
        + velocity_correlation_terms.b2
        * velocity_correlation_terms.d2
        * psi_c2_e2
    )


@validate_inputs(
    {
        "fluid_rms_velocity": "positive",
        "collisional_radius": "positive",
        "particle_inertia_time": "positive",
        "particle_velocity": "positive",
        "taylor_microscale": "positive",
        "eulerian_integral_length": "positive",
    }
)
def _compute_cross_correlation_velocity(
    fluid_rms_velocity: float,
    collisional_radius: Union[float, NDArray[np.float64]],
    particle_inertia_time: Union[float, NDArray[np.float64]],
    particle_velocity: Union[float, NDArray[np.float64]],
    taylor_microscale: float,
    eulerian_integral_length: float,
    velocity_correlation_terms: VelocityCorrelationTerms,
) -> Union[float, NDArray[np.float64]]:
    # pylint: disable=too-many-arguments, disable=too-many-positional-arguments
    """
    Compute the cross-correlation of the fluctuating velocities of droplets

    The function is given by:

        ⟨ v'^(1) v'^(2) ⟩ = (u'² f₂(R) / (τ_p1 τ_p2)) *\
                            [b₁ d₁ Φ(c₁, e₁) - b₁ d₂ Φ(c₁, e₂)\
                             - b₂ d₁ Φ(c₂, e₁) + b₂ d₂ Φ(c₂, e₂)]

    - u' (fluid_rms_velocity) : Fluid RMS fluctuation velocity [m/s].
    - τ_p1, τ_p2 : Inertia timescales of droplets 1 and 2 [s].
    - f₂(R) : Longitudinal velocity correlation function.
    - Φ(c, e) : Function Φ computed using `get_phi_ao2008`.

    Arguments:
    ----------
        - fluid_rms_velocity : Fluid RMS fluctuation velocity [m/s].
        - collisional_radius : Distance between two colliding droplets [m].
        - particle_inertia_time: Inertia timescale of droplet 1 [s].
        - particle_velocity: Droplet velocity [m/s].
        - taylor_microscale : Taylor microscale [m].
        - eulerian_integral_length : Eulerian integral length scale [m].
        - velocity_correlation_terms : Velocity correlation coefficients [-].

    Returns:
    --------
        - Cross-correlation velocity ⟨ v'^(1) v'^(2) ⟩ [m²/s²].

    References:
    -----------
    - Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on
        the geometric collision rate of sedimenting droplets. Part 2.
        Theory and parameterization. New Journal of Physics, 10.
        https://doi.org/10.1088/1367-2630/10/7/075016
    """
    f2_r = get_f2_longitudinal_velocity_correlation(
        collisional_radius, taylor_microscale, eulerian_integral_length
    )

    phi_c1_e1 = get_phi_ao2008(
        velocity_correlation_terms.c1,
        velocity_correlation_terms.e1,
        particle_inertia_time,
        particle_velocity,
    )
    phi_c1_e2 = get_phi_ao2008(
        velocity_correlation_terms.c1,
        velocity_correlation_terms.e2,
        particle_inertia_time,
        particle_velocity,
    )
    phi_c2_e1 = get_phi_ao2008(
        velocity_correlation_terms.c2,
        velocity_correlation_terms.e1,
        particle_inertia_time,
        particle_velocity,
    )
    phi_c2_e2 = get_phi_ao2008(
        velocity_correlation_terms.c2,
        velocity_correlation_terms.e2,
        particle_inertia_time,
        particle_velocity,
    )

    particle_inertia_time_pairwise_product = (
        particle_inertia_time[:, np.newaxis]
        * particle_inertia_time[np.newaxis, :]
    )

    return (
        (fluid_rms_velocity**2 * f2_r)
        / (particle_inertia_time_pairwise_product)
        * (
            velocity_correlation_terms.b1
            * velocity_correlation_terms.d1
            * phi_c1_e1
            - velocity_correlation_terms.b1
            * velocity_correlation_terms.d2
            * phi_c1_e2
            - velocity_correlation_terms.b2
            * velocity_correlation_terms.d1
            * phi_c2_e1
            + velocity_correlation_terms.b2
            * velocity_correlation_terms.d2
            * phi_c2_e2
        )
    )
