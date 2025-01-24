"""
Compute the function Φ(α, φ) for the given particle properties.
"""

from typing import Union
import numpy as np
from numpy.typing import NDArray

from particula.util.validate_inputs import validate_inputs


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
    """
    Compute the function Φ(α, φ) for the given particle properties.

    The function Φ(α, φ), when vₚ₁>vₚ₂, is defined as:

        Φ(α, φ) =
        {  1 / ( (vₚ₂ / φ) - (1 / τₚ₂) - (1 / α) )\
        -  1 / ( (vₚ₁ / φ) + (1 / τₚ₁) + (1 / α) ) }\
        ×  ( vₚ₁ - vₚ₂ ) / ( 2 φ ( (vₚ₁ - vₚ₂) / φ + (1 / τₚ₁) + (1 / τₚ₂) )² )

        + {  4 / ( (vₚ₂ / φ)² - ( (1 / τₚ₂) + (1 / α) )² )\
        -  1 / ( (vₚ₂ / φ) + (1 / τₚ₂) + (1 / α) )²\
        -  1 / ( (vₚ₂ / φ) - (1 / τₚ₂) - (1 / α) )²  }\
        ×  ( vₚ₂ / ( 2 φ ( (1 / τₚ₁) - (1 / α) \
        + ( (1 / τₚ₂) + (1 / α) ) (vₚ₁ / vₚ₂) ) ) )

        + {  2φ / ( (vₚ₁ / φ) + (1 / τₚ₁) + (1 / α) )\
        -  2φ / ( (vₚ₂ / φ) - (1 / τₚ₂) - (1 / α) )\
        -  vₚ₁ / ( ( (vₚ₁ / φ) + (1 / τₚ₁) + (1 / α) )² )\
        +  vₚ₂ / ( ( (vₚ₂ / φ) - (1 / τₚ₂) - (1 / α) )² )  }\
        ×  1 / ( 2φ ( (vₚ₁ - vₚ₂) / φ + (1 / τₚ₁) + (1 / τₚ₂) ) )

    Arguments:
    ----------
        - alpha : A parameter related to turbulence and droplet interactions
            [-].
        - phi : A characteristic velocity or timescale parameter [m/s].
        - particle_inertia_time : Inertia timescale of particle 1 τₚ₁,
            particle 2 τₚ₂ [s].
        - particle_velocity : Velocity of particle 1 vₚ₁,
            particle 2 vₚ₂ [m/s].

    Returns:
    --------
        - Φ(α, φ) value [-].

    References:
    -----------
    - Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on
        the geometric collision rate of sedimenting droplets. Part 2.
        Theory and parameterization. New Journal of Physics, 10.
        https://doi.org/10.1088/1367-2630/10/7/075016
    """
    # valid for v1 > v2, in pairwise comparison
    v1 = np.maximum(
        particle_velocity[:, np.newaxis], particle_velocity[np.newaxis, :]
    )
    v2 = np.minimum(
        particle_velocity[:, np.newaxis], particle_velocity[np.newaxis, :]
    )
    # tau1 > tau2 due to v1=tau1*gravity and v2=tau2*gravity
    tau1 = np.maximum(
        particle_inertia_time[:, np.newaxis],
        particle_inertia_time[np.newaxis, :],
    )
    tau2 = np.minimum(
        particle_inertia_time[:, np.newaxis],
        particle_inertia_time[np.newaxis, :],
    )

    term1 = _compute_phi_term1(v1, v2, tau1, tau2, alpha, phi)
    term2 = _compute_phi_term2(v1, v2, tau1, tau2, alpha, phi)
    term3 = _compute_phi_term3(v1, v2, tau1, tau2, alpha, phi)

    return term1 + term2 + term3


def _compute_phi_term1(
    v1: NDArray[np.float64],
    v2: NDArray[np.float64],
    tau1: NDArray[np.float64],
    tau2: NDArray[np.float64],
    alpha: NDArray[np.float64],
    phi: NDArray[np.float64],
) -> NDArray[np.float64]:
    # pylint: disable=too-many-arguments, too-many-positional-arguments
    """Compute the first term of the Φ function."""
    denominator1 = v2 / phi - (1 / tau2) - (1 / alpha)
    denominator2 = v1 / phi + (1 / tau1) + (1 / alpha)

    first_term = (1 / denominator1) - (1 / denominator2)

    common_denominator = (v1 - v2) / phi + (1 / tau1) + (1 / tau2)
    common_denominator_sq = common_denominator**2

    return first_term * ((v1 - v2) / (2 * phi * common_denominator_sq))


def _compute_phi_term2(
    v1: NDArray[np.float64],
    v2: NDArray[np.float64],
    tau1: NDArray[np.float64],
    tau2: NDArray[np.float64],
    alpha: NDArray[np.float64],
    phi: NDArray[np.float64],
) -> NDArray[np.float64]:
    # pylint: disable=too-many-arguments, too-many-positional-arguments
    """Compute the second term of the Φ function."""
    denominator1 = (v2 / phi + (1 / tau2) + (1 / alpha)) ** 2
    denominator2 = (v2 / phi - (1 / tau2) - (1 / alpha)) ** 2
    denominator3 = (v2 / phi) ** 2 - ((1 / tau2) + (1 / alpha)) ** 2

    second_term = (4 / denominator3) - (1 / denominator1) - (1 / denominator2)

    shared_denominator = (
        (1 / tau1) - (1 / alpha) + ((1 / tau2) + (1 / alpha)) * (v1 / v2)
    )

    return second_term * (v2 / (2 * phi * shared_denominator))


def _compute_phi_term3(
    v1: NDArray[np.float64],
    v2: NDArray[np.float64],
    tau1: NDArray[np.float64],
    tau2: NDArray[np.float64],
    alpha: NDArray[np.float64],
    phi: NDArray[np.float64],
) -> NDArray[np.float64]:
    # pylint: disable=too-many-arguments, too-many-positional-arguments
    """Compute the third term of the Φ function."""
    denominator1 = (v1 / phi) + (1 / tau1) + (1 / alpha)
    denominator2 = (v2 / phi) - (1 / tau2) - (1 / alpha)

    first_component = (2 * phi / denominator1) - (2 * phi / denominator2)
    second_component = -(v1 / denominator1**2) + (v2 / denominator2**2)

    shared_denominator = ((v1 - v2) / phi) + (1 / tau1) + (1 / tau2)

    return (first_component + second_component) / (
        2 * phi * shared_denominator
    )
