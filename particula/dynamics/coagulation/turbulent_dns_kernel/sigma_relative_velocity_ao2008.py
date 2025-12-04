"""Compute RMS fluctuation velocities and cross-correlations.

This module provides functions to compute the square of the RMS fluctuation
velocity and the cross-correlation of the fluctuating velocities for colliding
droplets, based on the theory of turbulent DNS kernels.
"""

from typing import NamedTuple, Union

import numpy as np
from numpy.typing import NDArray

from particula.util.validate_inputs import validate_inputs

from .phi_ao2008 import get_phi_ao2008
from .psi_ao2008 import get_psi_ao2008
from .velocity_correlation_f2_ao2008 import (
    get_f2_longitudinal_velocity_correlation,
)
from .velocity_correlation_terms_ao2008 import (
    compute_b1,
    compute_b2,
    compute_beta,
    compute_c1,
    compute_c2,
    compute_d1,
    compute_d2,
    compute_e1,
    compute_e2,
    compute_z,
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
    """Compute the variance of particle relative-velocity fluctuations.

    This function calculates the variance of particle relative-velocity
    fluctuations using the following equation:

    Where the equation is:

    - σ² = ⟨(v'²)⟩₁ + ⟨(v'²)⟩₂ - 2⟨v'¹ v'²⟩
        - v'¹, v'² are the fluctuating velocities for droplets 1 and 2.

    Arguments:
        - fluid_rms_velocity : Fluid RMS fluctuation velocity [m/s].
        - collisional_radius : Distance between two colliding droplets [m].
        - particle_inertia_time : Inertia timescale of droplet 1 [s].
        - particle_velocity : Droplet velocity [m/s].
        - taylor_microscale : Taylor microscale [m].
        - eulerian_integral_length : Eulerian integral length scale [m].
        - lagrangian_integral_time : Lagrangian integral time scale [s].

    Returns:
        - σ² : Variance of the particle relative-velocity fluctuation [m²/s²].

    Examples:
        ```py
        import numpy as np
        sigma_sq = get_relative_velocity_variance(
            fluid_rms_velocity=0.3,
            collisional_radius=np.array([1e-4, 2e-4]),
            particle_inertia_time=np.array([1.0, 1.2]),
            particle_velocity=np.array([0.1, 0.2]),
            taylor_microscale=0.01,
            eulerian_integral_length=0.1,
            lagrangian_integral_time=0.5,
            lagrangian_taylor_microscale_time=0.05
        )
        # Output: array([...])
        ```

    References:
        - Ayala, O. et al. (2008). Effects of turbulence on the geometric
          collision rate of sedimenting droplets. Part 2. Theory and
          parameterization. New Journal of Physics, 10.
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
        fluid_rms_velocity,
        particle_inertia_time,
        particle_velocity,
        vel_corr_terms,
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

    # Type narrowing: ensure array for indexing operations
    rms_array = (
        rms_velocity
        if isinstance(rms_velocity, np.ndarray)
        else np.array([rms_velocity])
    )

    return (
        rms_array[:, np.newaxis]
        + rms_array[np.newaxis, :]
        - 2 * cross_correlation
    )


@validate_inputs(
    {
        "fluid_rms_velocity": "positive",
        "particle_inertia_time": "positive",
    }
)
def _compute_rms_fluctuation_velocity(
    fluid_rms_velocity: float,
    particle_inertia_time: Union[float, NDArray[np.float64]],
    particle_velocity: Union[float, NDArray[np.float64]],
    velocity_correlation_terms: VelocityCorrelationTerms,
) -> Union[float, NDArray[np.float64]]:
    """Compute RMS fluctuation velocity for the k-th droplet.

    This function calculates the square of the RMS fluctuation velocity for
    the k-th droplet using the following equation:

    Where the equation is:

    - ⟨(v'ᵏ)²⟩ = (u'² / τ_pk) * [b₁ d₁ Ψ(c₁, e₁) - b₁ d₂ Ψ(c₁, e₂)
      - b₂ d₁ Ψ(c₂, e₁) + b₂ d₂ Ψ(c₂, e₂)]
        - v'ᵏ is the fluctuating velocity for droplet k.
        - u' (fluid_rms_velocity) : Fluid RMS fluctuation velocity [m/s].
        - τ_pk (particle_inertia_time) : Inertia timescale of the droplet k
            [s].
        - Ψ(c, e) : Function Ψ computed using `get_psi_ao2008`.

    Arguments:
        - fluid_rms_velocity : Fluid RMS fluctuation velocity [m/s].
        - particle_inertia_time : Inertia timescale of the droplet k [s].
        - particle_velocity : Droplet velocity [m/s].
        - velocity_correlation_terms : Velocity correlation coefficients [-].

    Returns:
        - RMS fluctuation velocity [m²/s²].

    Examples:
        ```py
        import numpy as np
        rms_fluct = _compute_rms_fluctuation_velocity(
            fluid_rms_velocity=0.3,
            particle_inertia_time=np.array([1.0, 1.2]),
            particle_velocity=np.array([0.1, 0.2]),
            velocity_correlation_terms=VelocityCorrelationTerms(
                b1=0.1, b2=0.2, d1=0.3, d2=0.4, c1=0.5, c2=0.6, e1=0.7, e2=0.8
            )
        )
        # Output: array([...])
        ```

    References:
        - Ayala, O. et al. (2008). Effects of turbulence on the geometric
          collision rate of sedimenting droplets. Part 2. Theory and
          parameterization. New Journal of Physics, 10.
          https://doi.org/10.1088/1367-2630/10/7/075016
    """
    psi_c1_e1 = get_psi_ao2008(
        velocity_correlation_terms.c1,
        velocity_correlation_terms.e1,
        particle_inertia_time,
        particle_velocity,
    )
    psi_c1_e2 = get_psi_ao2008(
        velocity_correlation_terms.c1,
        velocity_correlation_terms.e2,
        particle_inertia_time,
        particle_velocity,
    )
    psi_c2_e1 = get_psi_ao2008(
        velocity_correlation_terms.c2,
        velocity_correlation_terms.e1,
        particle_inertia_time,
        particle_velocity,
    )
    psi_c2_e2 = get_psi_ao2008(
        velocity_correlation_terms.c2,
        velocity_correlation_terms.e2,
        particle_inertia_time,
        particle_velocity,
    )

    return (fluid_rms_velocity**2 / particle_inertia_time) * (
        (
            velocity_correlation_terms.b1
            * velocity_correlation_terms.d1
            * psi_c1_e1
        )
        - (
            velocity_correlation_terms.b1
            * velocity_correlation_terms.d2
            * psi_c1_e2
        )
        - (
            velocity_correlation_terms.b2
            * velocity_correlation_terms.d1
            * psi_c2_e1
        )
        + (
            velocity_correlation_terms.b2
            * velocity_correlation_terms.d2
            * psi_c2_e2
        )
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
    """Compute cross-correlation of fluctuating velocities for two droplets.

    This function calculates the cross-correlation of the fluctuating
    velocities of two droplets using the following equation:

    Where the equation is

    - ⟨v'¹ v'²⟩ = (u'² f₂(R) / (τ_p1 τ_p2)) *
        [b₁ d₁ Φ(c₁, e₁) - b₁ d₂ Φ(c₁, e₂) - b₂ d₁ Φ(c₂, e₁) + b₂ d₂ Φ(c₂, e₂)]
        - v'¹, v'² are the fluctuating velocities for droplets 1 and 2.
        - u' (fluid_rms_velocity) : Fluid RMS fluctuation velocity [m/s].
        - τ_p1, τ_p2 : Inertia timescales of droplets 1 and 2 [s].
        - f₂(R) : Longitudinal velocity correlation function.
        - Φ(c, e) : Function Φ computed using `get_phi_ao2008`.

    Arguments:
        - fluid_rms_velocity : Fluid RMS fluctuation velocity [m/s].
        - collisional_radius : Distance between two colliding droplets [m].
        - particle_inertia_time : Inertia timescale of droplet 1 [s].
        - particle_velocity : Droplet velocity [m/s].
        - taylor_microscale : Taylor microscale [m].
        - eulerian_integral_length : Eulerian integral length scale [m].
        - velocity_correlation_terms : Velocity correlation coefficients [-].

    Returns:
        - Cross-correlation velocity [m²/s²].

    Examples:
        ```py
        import numpy as np
        ccv = _compute_cross_correlation_velocity(
            fluid_rms_velocity=0.3,
            collisional_radius=np.array([1e-4, 2e-4]),
            particle_inertia_time=np.array([1.0, 1.2]),
            particle_velocity=np.array([0.1, 0.2]),
            taylor_microscale=0.01,
            eulerian_integral_length=0.1,
            velocity_correlation_terms=VelocityCorrelationTerms(
                b1=0.1, b2=0.2, d1=0.3, d2=0.4, c1=0.5, c2=0.6, e1=0.7, e2=0.8
            )
        )
        # Output: array([...])
        ```

    References:
        - Ayala, O. et al. (2008). Effects of turbulence on the geometric
          collision rate of sedimenting droplets. Part 2. Theory and
          parameterization. New Journal of Physics, 10.
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

    # Type narrowing: ensure array for indexing operations
    inertia_array = (
        particle_inertia_time
        if isinstance(particle_inertia_time, np.ndarray)
        else np.array([particle_inertia_time])
    )

    particle_inertia_time_pairwise_product = (
        inertia_array[:, np.newaxis] * inertia_array[np.newaxis, :]
    )

    return (
        (fluid_rms_velocity**2 * f2_r)
        / (particle_inertia_time_pairwise_product)
        * (
            (
                velocity_correlation_terms.b1
                * velocity_correlation_terms.d1
                * phi_c1_e1
            )
            - (
                velocity_correlation_terms.b1
                * velocity_correlation_terms.d2
                * phi_c1_e2
            )
            - (
                velocity_correlation_terms.b2
                * velocity_correlation_terms.d1
                * phi_c2_e1
            )
            + (
                velocity_correlation_terms.b2
                * velocity_correlation_terms.d2
                * phi_c2_e2
            )
        )
    )
