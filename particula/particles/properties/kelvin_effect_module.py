"""Module to calculate the Kelvin effect on vapor pressure."""

from typing import Union

import numpy as np
from numpy.typing import NDArray

from particula.util.constants import GAS_CONSTANT
from particula.util.machine_limit import get_safe_exp
from particula.util.validate_inputs import validate_inputs

# Maximum Kelvin ratio to prevent overflow in exponential calculation
# exp(100) ≈ 2.7e43 is extremely large but numerically stable
# For particles smaller than ~0.1 nm, continuum mechanics breaks down
MAX_KELVIN_RATIO = 100.0


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
    """Compute the Kelvin radius (rₖ) to account for curvature effects on vapor
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
    """Compute the Kelvin exponential term to account for curvature effects.

    The Kelvin term (K) is given by:

    - K = exp(rₖ / rₚ)
        - K is dimensionless.
        - rₖ is the Kelvin radius (m).
        - rₚ is the particle radius (m).

    For very small particles (< 1 nm), the ratio rₖ / rₚ can become
    extremely large, leading to numerical overflow. To prevent this, the
    ratio is clipped to a maximum value of 100, corresponding to a Kelvin
    term of ~2.7e43. This is physically unrealistic but ensures numerical
    stability. Below ~0.1 nm, continuum mechanics breaks down anyway, so
    the condensation equations become questionable.

    Arguments:
        - particle_radius : Radius of the particle (m).
        - kelvin_radius_value : Precomputed Kelvin radius (m).

    Returns:
        - Dimensionless exponential factor adjusting vapor pressure.
          For extreme cases, the value is clipped to prevent overflow.

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

    # Suppress divide-by-zero warnings - zero radius is handled by clipping
    with np.errstate(divide="ignore", invalid="ignore"):
        if isinstance(particle_radius, np.ndarray) and not kelvin_expand:
            kelvin_ratio = kelvin_radius_value / particle_radius
            kelvin_ratio = np.clip(kelvin_ratio, None, MAX_KELVIN_RATIO)
            return get_safe_exp(kelvin_ratio)
        if (
            isinstance(particle_radius, np.ndarray)
            and (particle_radius.size > 1)
            and kelvin_expand
        ):
            particle_radius = particle_radius[:, np.newaxis]
            kelvin_ratio = kelvin_radius_value / particle_radius
            kelvin_ratio = np.clip(kelvin_ratio, None, MAX_KELVIN_RATIO)
            return get_safe_exp(kelvin_ratio)

        # Scalar case
        kelvin_ratio = kelvin_radius_value / particle_radius  # type: ignore[assignment]
        kelvin_ratio = np.clip(kelvin_ratio, None, MAX_KELVIN_RATIO)
        return get_safe_exp(kelvin_ratio)
