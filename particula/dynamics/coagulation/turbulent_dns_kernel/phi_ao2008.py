"""Compute the function Φ(α, φ) for the given particle properties."""

from typing import NamedTuple, Union

import numpy as np
from numpy.typing import NDArray

from particula.util.validate_inputs import validate_inputs


class PhiComputeTerms(NamedTuple):
    """Parameters for computing Φ function terms."""

    v1: Union[float, NDArray[np.float64]]
    v2: Union[float, NDArray[np.float64]]
    tau1: Union[float, NDArray[np.float64]]
    tau2: Union[float, NDArray[np.float64]]
    alpha: Union[float, NDArray[np.float64]]
    phi: Union[float, NDArray[np.float64]]


@validate_inputs(
    {
        "alpha": "positive",
        "phi": "positive",
        "particle_inertia_time": "positive",
        "particle_velocity": "positive",
    }
)
def get_phi_ao2008(
    alpha: Union[float, NDArray[np.float64]],
    phi: Union[float, NDArray[np.float64]],
    particle_inertia_time: Union[float, NDArray[np.float64]],
    particle_velocity: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """Compute the function Φ(α, φ) for the given particle properties using
    Ayala et al. (2008).

    This function calculates Φ(α, φ) when vₚ₁ > vₚ₂ by considering the
    velocities (vₚ₁, vₚ₂) and inertia times (τₚ₁, τₚ₂). The equation is:

    Φ(α, φ), for vₚ₁ > vₚ₂ =
        {  1 / ( (vₚ₂ / φ) - (1 / τₚ₂) - (1 / α) )\
        -  1 / ( (vₚ₁ / φ) + (1 / τₚ₁) + (1 / α) ) }\
        ×  ( vₚ₁ - vₚ₂ ) / ( 2 φ ( (vₚ₁ - vₚ₂ / φ) + (1 / τₚ₁) + (1 / τₚ₂) )² )

        + {  4 / ( (vₚ₂ / φ)² - ( (1 / τₚ₂) + (1 / α) )² )\
        -  1 / ( (vₚ₂ / φ) + (1 / τₚ₂) + (1 / α) )²\
        -  1 / ( (vₚ₂ / φ) - (1 / τₚ₂) - (1 / α) )²  }\
        ×  ( vₚ₂ / ( 2 φ ( (1 / τₚ₁) - (1 / α) \
        + ( (1 / τₚ₂) + (1 / α) ) (vₚ₁ / vₚ₂) ) ) )

        + {  2φ / ( (vₚ₁ / φ) + (1 / τₚ₁) + (1 / α) )\
        -  2φ / ( (vₚ₂ / φ) - (1 / τₚ₂) - (1 / α) )\
        -  vₚ₁ / ( ( (vₚ₁ / φ) + (1 / τₚ₁) + (1 / α) )² )\
        +  vₚ₂ / ( ( (vₚ₂ / φ) - (1 / τₚ₂) - (1 / α) )² )  }\
        ×  1 / ( 2φ ( (vₚ₁ - vₚ₂ / φ) + (1 / τₚ₁) + (1 / τₚ₂) ) )

      - v₁ and v₂: Velocities of particles 1 and 2 in m/s.
      - τ₁ and τ₂: Inertia timescales of particles 1 and 2 in s.
      - α: Turbulent interaction parameter (dimensionless).
      - φ: Characteristic velocity (m/s).

    Arguments:
        - alpha : Turbulence/droplet interaction parameter (dimensionless).
        - phi : Characteristic velocity parameter (m/s).
        - particle_inertia_time : Inertia timescales τₚ₁ and τₚ₂ (s).
        - particle_velocity : Velocities vₚ₁ and vₚ₂ (m/s).

    Returns:
        - The computed Φ(α, φ) (dimensionless).

    Examples:
        ```py
        import numpy as np
        from particula.dynamics.coagulation.turbulent_dns_kernel.phi_ao2008
            import get_phi_ao2008

        alpha_val = 0.3
        phi_val = 0.1
        inertia_times = np.array([0.05, 0.06])
        velocities = np.array([0.2, 0.18])
        result = get_phi_ao2008(alpha_val, phi_val, inertia_times, velocities)
        print(result)
        ```

    References:
        - Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on
          the geometric collision rate of sedimenting droplets. Part 2.
          Theory and parameterization. New Journal of Physics, 10.
          https://doi.org/10.1088/1367-2630/10/7/075016
    """
    # valid for v1 > v2, in pairwise comparison
    # Type narrowing: ensure arrays for indexing operations
    velocity_array = (
        particle_velocity
        if isinstance(particle_velocity, np.ndarray)
        else np.array([particle_velocity])
    )
    inertia_array = (
        particle_inertia_time
        if isinstance(particle_inertia_time, np.ndarray)
        else np.array([particle_inertia_time])
    )

    v1 = np.maximum(
        velocity_array[:, np.newaxis], velocity_array[np.newaxis, :]
    )
    v2 = np.minimum(
        velocity_array[:, np.newaxis], velocity_array[np.newaxis, :]
    )
    # tau1 > tau2 due to v1~=tau1*gravity and v2~=tau2*gravity
    tau1 = np.maximum(
        inertia_array[:, np.newaxis],
        inertia_array[np.newaxis, :],
    )
    tau2 = np.minimum(
        inertia_array[:, np.newaxis],
        inertia_array[np.newaxis, :],
    )

    phi_compute_terms = PhiComputeTerms(v1, v2, tau1, tau2, alpha, phi)

    term1 = _compute_phi_term1(phi_compute_terms)
    term2 = _compute_phi_term2(phi_compute_terms)
    term3 = _compute_phi_term3(phi_compute_terms)

    return term1 + term2 + term3


def _compute_phi_term1(
    terms: PhiComputeTerms,
) -> Union[float, NDArray[np.float64]]:
    """Compute the first term of the Φ function.

     term_1 = {
        1 / ( (vₚ₂ / φ) - (1 / τₚ₂) - (1 / α) )\
        -  1 / ( (vₚ₁ / φ) + (1 / τₚ₁) + (1 / α) )
    }\
        ×  ( vₚ₁ - vₚ₂ ) / ( 2 φ ( (vₚ₁ - vₚ₂ / φ) + (1 / τₚ₁) + (1 / τₚ₂) )² )
    """
    denominator1 = terms.v2 / terms.phi - (1 / terms.tau2) - (1 / terms.alpha)
    denominator2 = terms.v1 / terms.phi + (1 / terms.tau1) + (1 / terms.alpha)

    first_term = (1 / denominator1) - (1 / denominator2)

    common_denominator = (
        (terms.v1 - terms.v2 / terms.phi) + (1 / terms.tau1) + (1 / terms.tau2)
    )

    return first_term * (
        (terms.v1 - terms.v2) / (2 * terms.phi * common_denominator**2)
    )


def _compute_phi_term2(
    terms: PhiComputeTerms,
) -> Union[float, NDArray[np.float64]]:
    """Compute the second term of the Φ function.

    term₂ =
    {
      4 / [ (v₂ / φ)² − ( (1 / τ₂) + (1 / α) )² ] \
      − 1 / [ (v₂ / φ) + (1 / τ₂) + (1 / α) ]² \
      − 1 / [ (v₂ / φ) − (1 / τ₂) − (1 / α) ]² \
    }
    × v₂ / \
      [ 2 φ ( (1 / τ₁) − (1 / α) + ( (1 / τ₂) + (1 / α) ) × (v₁ / v₂) ) ]
    """
    denominator1 = (terms.v2 / terms.phi) ** 2 - (
        (1 / terms.tau2) + (1 / terms.alpha)
    ) ** 2
    denominator2 = (
        terms.v2 / terms.phi + (1 / terms.tau2) + (1 / terms.alpha)
    ) ** 2
    denominator3 = (
        terms.v2 / terms.phi - (1 / terms.tau2) - (1 / terms.alpha)
    ) ** 2

    second_term = (4 / denominator1) - (1 / denominator2) - (1 / denominator3)

    shared_denominator = (
        (1 / terms.tau1)
        - (1 / terms.alpha)
        + ((1 / terms.tau2) + (1 / terms.alpha)) * (terms.v1 / terms.v2)
    )

    return second_term * (terms.v2 / (2 * terms.phi * shared_denominator))


def _compute_phi_term3(
    terms: PhiComputeTerms,
) -> Union[float, NDArray[np.float64]]:
    """Compute the third term of the Φ function.

    term_3 =
    {
        2φ / ( (vₚ₁ / φ) + (1 / τₚ₁) + (1 / α) )\
        -  2φ / ( (vₚ₂ / φ) - (1 / τₚ₂) - (1 / α) )\
        -  vₚ₁ / ( ( (vₚ₁ / φ) + (1 / τₚ₁) + (1 / α) )² )\
        +  vₚ₂ / ( ( (vₚ₂ / φ) - (1 / τₚ₂) - (1 / α) )² )
    }
        ×  1 / ( 2φ ( (vₚ₁ - vₚ₂/φ ) + (1 / τₚ₁) + (1 / τₚ₂) ) )
    """
    denominator1 = (terms.v1 / terms.phi) + (1 / terms.tau1) + (1 / terms.alpha)
    denominator2 = (terms.v2 / terms.phi) - (1 / terms.tau2) - (1 / terms.alpha)

    first_component = (2 * terms.phi / denominator1) - (
        2 * terms.phi / denominator2
    )
    second_component = -(terms.v1 / denominator1**2) + (
        terms.v2 / denominator2**2
    )

    shared_denominator = (
        (terms.v1 - terms.v2 / terms.phi) + (1 / terms.tau1) + (1 / terms.tau2)
    )

    return (first_component + second_component) / (
        2 * terms.phi * shared_denominator
    )
