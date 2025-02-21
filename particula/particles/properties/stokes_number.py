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
    Compute the Stokes number (St) to measure particle inertia relative to
    fluid flow.

    The Stokes number is a dimensionless parameter reflecting how much a
    particle resists following changes in the fluid’s motion. If St >> 1,
    particle inertia dominates; if St << 1, the particle closely follows
    fluid flow. Mathematically:

    - St = τ_p / τ_k
        - St : Stokes number (dimensionless),
        - τ_p : Particle inertia time [s],
        - τ_k : Kolmogorov timescale [s].

    Arguments:
        - particle_inertia_time : Particle inertia time in seconds (s).
        - kolmogorov_time : Kolmogorov timescale in seconds (s).

    Returns:
        - Dimensionless Stokes number (float or NDArray[np.float64]).

    Examples:
        ``` py title="Example"
        import particula as par
        par.particles.get_stokes_number(1e-3, 2e-3)
        # Output: 0.5
        ```

    References:
        - [Stokes number, Wikipedia](https://en.wikipedia.org/wiki/Stokes_number)
        - Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric Chemistry and
          Physics, 3rd ed., Wiley-Interscience.
    """
    return particle_inertia_time / kolmogorov_time
