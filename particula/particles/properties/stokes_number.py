"""
Stokes number calculation.
"""

from typing import Union
from numpy.typing import NDArray
import numpy as np

from particula.util.validate_inputs import validate_inputs


@validate_inputs(
    {
        "particle_inertia_time": "positive",
        "kolmogorov_time": "positive",
    }
)
def get_stokes_number(
    particle_inertia_time: Union[float, NDArray[np.float64]],
    kolmogorov_time: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate the Stokes number.

    The Stokes number (St) is a dimensionless [-] parameter representing the
    relative importance of particle inertia in a fluid flow. A high Stokes
    number indicates that a particle resists following fluid motion, while a
    low Stokes number means the particle closely follows the flow. Given by:

        St = τ_p / τ_k

    - St : Stokes number [-]
    - τ_p (particle_inertia_time) : Particle inertia time [s]
    - τ_k (kolmogorov_time) : Kolmogorov timescale of turbulence [s]

    Arguments:
    ----------
        - particle_inertia_time : Particle inertia time [s]
        - kolmogorov_time : Kolmogorov timescale [s]

    Returns:
    --------
        - Stokes number [-]

    References:
    -----------
        - Wikipedia https://en.wikipedia.org/wiki/Stokes_number
    """
    return particle_inertia_time / kolmogorov_time
