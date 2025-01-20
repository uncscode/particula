"""
Compute the square of the RMS fluctuation velocity and the cross-correlation
of the fluctuating velocities.
"""

from typing import Union
import numpy as np
from numpy.typing import NDArray

from particula.util.validate_inputs import validate_inputs
from particula.util.self_broadcast import (
    get_pairwise_product_matrix,
    get_pairwise_sum_matrix,
)
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


@validate_inputs(
    {
        "turbulence_intensity": "positive",
        "collisional_radius": "positive",
        "particle_inertia_time": "positive",
    }
)
def get_relative_velocity_variance(
    turbulence_intensity: Union[float, NDArray[np.float64]],
    collisional_radius: Union[float, NDArray[np.float64]],
    particle_inertia_time: Union[float, NDArray[np.float64]],
    particle_velocity: Union[float, NDArray[np.float64]],
    taylor_microscale: Union[float, NDArray[np.float64]],
    eulerian_integral_length: Union[float, NDArray[np.float64]],
    lagrangian_integral_time: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """
    Compute the variance of particle relative-velocity fluctuation.

    The function is given by:

        σ² = ⟨ (v'^(2))² ⟩ + ⟨ (v'^(1))² ⟩ - 2 ⟨ v'^(1) v'^(2) ⟩

    - ⟨ (v'^(k))² ⟩ is the square of the RMS fluctuation velocity for droplet k
    - ⟨ v'^(1) v'^(2) ⟩ is the cross-correlation of the fluctuating velocities.

    Arguments:
    ----------
        - turbulence_intensity : Fluid RMS fluctuation velocity [m/s].
        - collisional_radius : Distance between two colliding droplets [m].
        - particle_inertia_time : Inertia timescale of droplet 1 [s].

    Returns:
    --------
        - σ² : Variance of particle relative-velocity fluctuation [m²/s²].

    References:
    -----------
    - Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on
        the geometric collision rate of sedimenting droplets. Part 2.
        Theory and parameterization. New Journal of Physics, 10.
        https://doi.org/10.1088/1367-2630/10/7/075016
    """

    z = compute_z(lagrangian_integral_time, eulerian_integral_length)
    beta = compute_beta(taylor_microscale, eulerian_integral_length)
    b1 = compute_b1(z)
    b2 = compute_b2(z)
    d1 = compute_d1(beta)
    d2 = compute_d2(beta)
    c1 = compute_c1(z, lagrangian_integral_time)
    c2 = compute_c2(z, lagrangian_integral_time)
    e1 = compute_e1(z, eulerian_integral_length)
    e2 = compute_e2(z, eulerian_integral_length)

    rms_velocity = compute_rms_fluctuation_velocity(
        turbulence_intensity,
        particle_inertia_time,
        b1,
        b2,
        d1,
        d2,
        c1,
        c2,
        e1,
        e2,
    )

    cross_correlation = compute_cross_correlation_velocity(
        turbulence_intensity,
        collisional_radius,
        particle_inertia_time,
        particle_velocity,
        taylor_microscale,
        eulerian_integral_length,
        b1,
        b2,
        d1,
        d2,
        c1,
        c2,
        e1,
        e2,
    )

    return get_pairwise_sum_matrix(rms_velocity**2) - 2 * cross_correlation


@validate_inputs(
    {
        "turbulence_intensity": "positive",
        "particle_inertia_time": "positive",
    }
)
def compute_rms_fluctuation_velocity(
    turbulence_intensity: Union[float, NDArray[np.float64]],
    particle_inertia_time: Union[float, NDArray[np.float64]],
    b1: Union[float, NDArray[np.float64]],
    b2: Union[float, NDArray[np.float64]],
    d1: Union[float, NDArray[np.float64]],
    d2: Union[float, NDArray[np.float64]],
    c1: Union[float, NDArray[np.float64]],
    c2: Union[float, NDArray[np.float64]],
    e1: Union[float, NDArray[np.float64]],
    e2: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """
    Compute the square of the RMS fluctuation velocity for the k-th droplet.

    The function is given by:

        ⟨ (v'^(k))² ⟩ = (u'² / τ_pk) *\
                        [b₁ d₁ Ψ(c₁, e₁) - b₁ d₂ Ψ(c₁, e₂)\
                         - b₂ d₁ Ψ(c₂, e₁) + b₂ d₂ Ψ(c₂, e₂)]

    - u' (turbulence_intensity) : Fluid RMS fluctuation velocity [m/s].
    - τ_pk (particle_inertia_time) : Inertia timescale of the droplet k [s].
    - Ψ(c, e) : Function Ψ computed using `get_psi_ao2008`.

    Arguments:
    ----------
        - turbulence_intensity : Fluid RMS fluctuation velocity [m/s].
        - particle_inertia_time : Inertia timescale of the droplet k [s].
        - b1, b2 : Velocity correlation coefficients.
        - d1, d2 : Velocity correlation coefficients.
        - c1, c2 : Velocity correlation coefficients.
        - e1, e2 : Velocity correlation coefficients.

    Returns:
    --------
        - RMS fluctuation velocity squared ⟨ (v'^(k))² ⟩ [m²/s²].
    """
    psi_c1_e1 = get_psi_ao2008(
        c1, e1, particle_inertia_time, turbulence_intensity
    )
    psi_c1_e2 = get_psi_ao2008(
        c1, e2, particle_inertia_time, turbulence_intensity
    )
    psi_c2_e1 = get_psi_ao2008(
        c2, e1, particle_inertia_time, turbulence_intensity
    )
    psi_c2_e2 = get_psi_ao2008(
        c2, e2, particle_inertia_time, turbulence_intensity
    )

    return (turbulence_intensity**2 / particle_inertia_time) * (
        b1 * d1 * psi_c1_e1
        - b1 * d2 * psi_c1_e2
        - b2 * d1 * psi_c2_e1
        + b2 * d2 * psi_c2_e2
    )


@validate_inputs(
    {
        "turbulence_intensity": "positive",
        "collisional_radius": "positive",
        "particle_inertia_time_1": "positive",
        "particle_inertia_time_2": "positive",
    }
)
def compute_cross_correlation_velocity(
    turbulence_intensity: Union[float, NDArray[np.float64]],
    collisional_radius: Union[float, NDArray[np.float64]],
    particle_inertia_time: Union[float, NDArray[np.float64]],
    particle_velocity: Union[float, NDArray[np.float64]],
    taylor_microscale: Union[float, NDArray[np.float64]],
    eulerian_integral_length: Union[float, NDArray[np.float64]],
    b1: Union[float, NDArray[np.float64]],
    b2: Union[float, NDArray[np.float64]],
    d1: Union[float, NDArray[np.float64]],
    d2: Union[float, NDArray[np.float64]],
    c1: Union[float, NDArray[np.float64]],
    c2: Union[float, NDArray[np.float64]],
    e1: Union[float, NDArray[np.float64]],
    e2: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """
    Compute the cross-correlation of the fluctuating velocities of droplets

    The function is given by:

        ⟨ v'^(1) v'^(2) ⟩ = (u'² f₂(R) / (τ_p1 τ_p2)) *\
                            [b₁ d₁ Φ(c₁, e₁) - b₁ d₂ Φ(c₁, e₂)\
                             - b₂ d₁ Φ(c₂, e₁) + b₂ d₂ Φ(c₂, e₂)]

    - u' (turbulence_intensity) : Fluid RMS fluctuation velocity [m/s].
    - τ_p1, τ_p2 : Inertia timescales of droplets 1 and 2 [s].
    - f₂(R) : Longitudinal velocity correlation function.
    - Φ(c, e) : Function Φ computed using `get_phi_ao2008`.

    Arguments:
    ----------
        - turbulence_intensity : Fluid RMS fluctuation velocity [m/s].
        - collisional_radius : Distance between two colliding droplets [m].
        - particle_inertia_time: Inertia timescale of droplet 1 [s].
        - particle_velocity: Droplet velocity [m/s].
        - taylor_microscale : Taylor microscale [m].
        - eulerian_integral_length : Eulerian integral length scale [m].
        - b1, b2 : Velocity correlation coefficients.
        - d1, d2 : Velocity correlation coefficients.
        - c1, c2 : Velocity correlation coefficients.
        - e1, e2 : Velocity correlation coefficients.

    Returns:
    --------
        - Cross-correlation velocity ⟨ v'^(1) v'^(2) ⟩ [m²/s²].
    """
    f2_r = get_f2_longitudinal_velocity_correlation(
        collisional_radius, taylor_microscale, eulerian_integral_length
    )

    phi_c1_e1 = get_phi_ao2008(
        c1, e1, particle_inertia_time, particle_velocity
    )
    phi_c1_e2 = get_phi_ao2008(
        c1, e2, particle_inertia_time, particle_velocity
    )
    phi_c2_e1 = get_phi_ao2008(
        c2, e1, particle_inertia_time, particle_velocity
    )
    phi_c2_e2 = get_phi_ao2008(
        c2, e2, particle_inertia_time, particle_velocity
    )

    particle_inertia_time_pairwise_product = get_pairwise_product_matrix(
        particle_inertia_time
    )

    return (
        (turbulence_intensity**2 * f2_r)
        / (particle_inertia_time_pairwise_product)
        * (
            b1 * d1 * phi_c1_e1
            - b1 * d2 * phi_c1_e2
            - b2 * d1 * phi_c2_e1
            + b2 * d2 * phi_c2_e2
        )
    )
