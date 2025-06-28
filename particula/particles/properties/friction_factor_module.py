"""Module for friction factor of a particle in a fluid.

Zhang, C., Thajudeen, T., Larriba, C., Schwartzentruber, T. E., &
Hogan, C. J. (2012). Determination of the Scalar Friction Factor for
Nonspherical Particles and Aggregates Across the Entire Knudsen Number Range
by Direct Simulation Monte Carlo (DSMC). Aerosol Science and Technology,
46(10), 1065-1078. https://doi.org/10.1080/02786826.2012.690543
"""

from typing import Union

import numpy as np
from numpy.typing import NDArray

from particula.util.validate_inputs import validate_inputs


@validate_inputs(
    {
        "particle_radius": "nonnegative",
        "dynamic_viscosity": "positive",
        "slip_correction": "positive",
    }
)
def get_friction_factor(
    particle_radius: Union[float, NDArray[np.float64]],
    dynamic_viscosity: float,
    slip_correction: Union[float, NDArray[np.float64]],
):
    """Calculate the friction factor for a particle in a fluid.

    This friction factor (f) is the proportionality constant between
    the fluid velocity and the resulting drag force on the particle.
    The formula used is:

    - f = (6πμ r) / C
        - f is the friction factor (N·s/m),
        - μ is the dynamic viscosity of the fluid (Pa·s),
        - r is the radius of the particle (m),
        - C is the slip correction factor (dimensionless).

    Arguments:
        - particle_radius : Radius of the particle in meters (m).
        - dynamic_viscosity : Dynamic viscosity of the fluid in Pa·s.
        - slip_correction : Slip correction factor (dimensionless).

    Returns:
        - The friction factor of the particle in N·s/m.

    Examples:
        ``` py title="Example"
        import particula as par
        par.particles.get_friction_factor(
            particle_radius=1e-7,
            dynamic_viscosity=1.8e-5,
            slip_correction=1.1
        )
        # Output: ...
        ```

    References:
        - Zhang, C., Thajudeen, T., Larriba, C., Schwartzentruber, T. E.,
          & Hogan, C. J. (2012). "Determination of the Scalar Friction Factor
          for Nonspherical Particles and Aggregates Across the Entire Knudsen
          Number Range by Direct Simulation Monte Carlo (DSMC)."
          Aerosol Science and Technology, 46(10), 1065-1078.
          https://doi.org/10.1080/02786826.2012.690543
    """
    return 6 * np.pi * dynamic_viscosity * particle_radius / slip_correction
