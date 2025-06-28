"""Psi function for the droplet collision kernel in the turbulent DNS model."""

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
    """Compute the function Ψ(α, φ) for the k-th droplet.

    This function calculates Ψ(α, φ) for the droplet collision kernel in the
    turbulent DNS model. The equation is:

    - Ψ(α, φ) = 1 / ((1/τₚₖ) + (1/α) + (vₚₖ/φ))
                - (vₚₖ / (2φ ((1/τₚₖ) + (1/α) + (vₚₖ/φ))²))
        - τₚₖ is the inertia timescale of the droplet (s),
        - α is a parameter related to turbulence (dimensionless),
        - φ is a characteristic velocity/timescale parameter (m/s),
        - vₚₖ is the droplet velocity (m/s).

    Arguments:
        - alpha : Parameter related to turbulence (dimensionless).
        - phi : Characteristic velocity or timescale parameter (m/s).
        - particle_inertia_time : Inertia timescale of the droplet τₚₖ (s).
        - particle_velocity : Velocity of the droplet vₚₖ (m/s).

    Returns:
        - The value of Ψ(α, φ) (dimensionless).

    Examples:
        ``` py
        import numpy as np
        import particula as par

        alpha = 0.5
        phi = 0.2
        particle_inertia_time = 0.05
        particle_velocity = 0.3

        psi_value = par.dyanmics.get_psi_ao2008(
            alpha, phi, particle_inertia_time, particle_velocity
        )
        print(psi_value)
        ```

    References:
        - Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on
          the geometric collision rate of sedimenting droplets. Part 2.
          Theory and parameterization. New Journal of Physics, 10.
          https://doi.org/10.1088/1367-2630/10/7/075016
    """
    denominator = (
        (1 / particle_inertia_time) + (1 / alpha) + (particle_velocity / phi)
    )
    return 1 / denominator - (particle_velocity / (2 * phi * denominator**2))
