"""Module to calculate the Kelvin effect on vapor pressure."""

from typing import Union
from numpy.typing import NDArray
import numpy as np

from particula.util.constants import GAS_CONSTANT
from particula.util.machine_limit import safe_exp
from particula.util.validate_inputs import validate_inputs


@validate_inputs(
    {
        "effective_surface_tension": "positive",
        "effective_density": "positive",
        "molar_mass": "positive",
        "temperature": "positive",
    }
)
def get_kelvin_radius(
    effective_surface_tension: Union[float, NDArray[np.float64]],
    effective_density: Union[float, NDArray[np.float64]],
    molar_mass: Union[float, NDArray[np.float64]],
    temperature: float,
) -> Union[float, NDArray[np.float64]]:
    """
    Compute the Kelvin radius (rₖ) to account for curvature effects on vapor
    pressure.

    The Kelvin radius is defined by:

    - rₖ = (2 × σ × M) / (R × T × ρ)
        - rₖ is Kelvin radius in meters (m).
        - σ is the effective surface tension in N/m.
        - M is the molar mass in kg/mol.
        - R is the universal gas constant in J/(mol·K).
        - T is the temperature in Kelvin (K).
        - ρ is the effective density in kg/m³.

    Arguments:
        - effective_surface_tension : Surface tension of the mixture (N/m).
        - effective_density : Effective density of the mixture (kg/m³).
        - molar_mass : Molar mass (kg/mol).
        - temperature : Temperature of the system (K).

    Returns:
        - Kelvin radius in meters (float or NDArray[np.float64]).

    Examples:
        ``` py title="Example"
        import numpy as np
        import particula as par
        par.particles.get_kelvin_radius(
            effective_surface_tension=0.072,
            effective_density=1000.0,
            molar_mass=0.018,
            temperature=298.15
        )
        # Output: ...
        ```

    References:
        - "Kelvin equation," Wikipedia,
          https://en.wikipedia.org/wiki/Kelvin_equation
    """
    return (2 * effective_surface_tension * molar_mass) / (
        GAS_CONSTANT * temperature * effective_density
    )


@validate_inputs(
    {
        "particle_radius": "nonnegative",
        "kelvin_radius_value": "nonnegative",
    }
)
def get_kelvin_term(
    particle_radius: Union[float, NDArray[np.float64]],
    kelvin_radius_value: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """
    Compute the Kelvin exponential term to account for curvature effects.

    The Kelvin term (K) is given by:

    - K = exp(rₖ / rₚ)
        - K is dimensionless.
        - rₖ is the Kelvin radius (m).
        - rₚ is the particle radius (m).

    Arguments:
        - particle_radius : Radius of the particle (m).
        - kelvin_radius_value : Precomputed Kelvin radius (m).

    Returns:
        - Dimensionless exponential factor adjusting vapor pressure.

    Examples:
        ``` py title="Example"
        import particula as par
        par.particles.get_kelvin_term(
            particle_radius=1e-7,
            kelvin_radius_value=2e-7
        )
        print(kv_term)
        # Output: ...
        ```

    References:
        - Donahue, N. M., et al. (2013). "How do organic vapors contribute to
          new-particle formation?" Faraday Discussions, 165, 91–104.
          https://doi.org/10.1039/C3FD00046J. [check]
    """
    kelvin_expand = False
    # Broadcast the arrays if necessary np.isscalar(kelvin_radius_value)
    if isinstance(kelvin_radius_value, np.ndarray) and (
        kelvin_radius_value.size > 1
    ):
        kelvin_expand = True
        kelvin_radius_value = kelvin_radius_value[np.newaxis, :]
    if isinstance(particle_radius, np.ndarray) and not kelvin_expand:
        return safe_exp(kelvin_radius_value / particle_radius)
    if (
        isinstance(particle_radius, np.ndarray)
        and (particle_radius.size > 1)
        and kelvin_expand
    ):
        particle_radius = particle_radius[:, np.newaxis]
    return safe_exp(kelvin_radius_value / particle_radius)
