"""
Psi function for the droplet collision kernel in the turbulent DNS model.
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
def get_psi_ao2008(
    alpha: Union[float, NDArray[np.float64]],
    phi: Union[float, NDArray[np.float64]],
    particle_inertia_time: Union[float, NDArray[np.float64]],
    particle_velocity: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """
    Compute the function Ψ(α, φ) for the k-th droplet.

    The function Ψ(α, φ) is defined as:

        Ψ(α, φ) = 1 / ((1/τ_pk) + (1/α) + (v_pk/φ))
                - (v_pk / (2φ ((1/τ_pk) + (1/α) + (v_pk/φ))^2))

    - τ_pk (particle_inertia_time) : Inertia timescale of the k-th droplet
        [s].
    - α : A parameter related to turbulence and droplet interactions [-].
    - φ : A characteristic velocity or timescale parameter [m/s].
    - v_pk (particle_velocity) : Velocity of the k-th droplet [m/s].

    Arguments:
    ----------
        - alpha : A parameter related to turbulence and droplet interactions
            [-].
        - phi : A characteristic velocity or timescale parameter [m/s].
        - particle_inertia_time : Inertia timescale of the droplet τ_pk [s].
        - particle_velocity : Velocity of the droplet v_pk [m/s].

    Returns:
    --------
        - Ψ(α, φ) value [-].

    References:
    -----------
    - Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on
        the geometric collision rate of sedimenting droplets. Part 2.
        Theory and parameterization. New Journal of Physics, 10.
        https://doi.org/10.1088/1367-2630/10/7/075016
    """
    denominator = (
        (1 / particle_inertia_time) + (1 / alpha) + (particle_velocity / phi)
    )
    return (
        1 / denominator
        - (particle_velocity / (2 * phi * denominator**2))
    )
