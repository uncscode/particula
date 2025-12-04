"""Module for the diffusive knudsen number."""

from typing import Union

import numpy as np
from numpy.typing import NDArray

from particula.particles.properties import coulomb_enhancement
from particula.util.constants import BOLTZMANN_CONSTANT
from particula.util.reduced_quantity import get_reduced_self_broadcast
from particula.util.validate_inputs import validate_inputs


@validate_inputs(
    {
        "particle_radius": "nonnegative",
        "particle_mass": "nonnegative",
        "friction_factor": "nonnegative",
    }
)
def get_diffusive_knudsen_number(
    particle_radius: Union[float, NDArray[np.float64]],
    particle_mass: Union[float, NDArray[np.float64]],
    friction_factor: Union[float, NDArray[np.float64]],
    coulomb_potential_ratio: Union[float, NDArray[np.float64]] = 0.0,
    temperature: float = 298.15,
) -> Union[float, NDArray[np.float64]]:
    """Compute the diffusive Knudsen number for particle-particle interactions.

    The *diffusive* Knudsen number (Kn_d) differs from the standard Knudsen
    number. It represents the ratio of the mean particle persistence
    distance to the effective Coulombic interaction scale. Mathematically:

    - Kn_d = [ √(k_B × T × μ_red) / f_red ] / [ (rᵢ + rⱼ) × (Γ_c / Γ_k) ]
        - k_B is the Boltzmann constant (J/K).
        - T is the temperature (K).
        - μ_red is the reduced mass of particles (kg).
        - f_red is the reduced friction factor (dimensionless).
        - rᵢ + rⱼ is the sum of radii for the interacting particles (m).
        - Γ_c is the continuum-limit Coulomb enhancement factor(dimensionless).
        - Γ_k is the kinetic-limit Coulomb enhancement factor (dimensionless).

    Arguments:
        - particle_radius : Radius of the particle(s) in meters (m).
        - particle_mass : Mass of the particle(s) in kilograms (kg).
        - friction_factor : Friction factor(s) (dimensionless).
        - coulomb_potential_ratio : Coulomb potential ratio (dimensionless),
          zero if no charge.
        - temperature : Temperature of the system in Kelvin (K).

    Returns:
        - The diffusive Knudsen number, either a float or NDArray[np.float64].

    Examples:
        ```py title="Single Particle Example"
        import numpy as np
        import particula as par
        par.particles.get_diffusive_knudsen_number(
            particle_radius=1e-7,
            particle_mass=1e-17,
            friction_factor=0.8,
            coulomb_potential_ratio=0.3,
            temperature=300
        )
        # Output: 0.12...
        ```
        ```py title="Multiple Particles Example"
        import numpy as np
        import particula as par
        # Multiple particles example
        radius_arr = np.array([1e-7, 2e-7])
        mass_arr = np.array([1e-17, 2e-17])
        friction_arr = np.array([0.8, 1.1])
        potential_arr = np.array([0.3, 0.5])
        par.particles.par.get_diffusive_knudsen_number(
            radius_arr, mass_arr, friction_arr, potential_arr
        )
        # Output: array([...])
        ```

    References:
        - Chahl, H. S., & Gopalakrishnan, R. (2019). "High potential, near free
          molecular regime Coulombic collisions in aerosols and dusty plasmas."
          Aerosol Science and Technology, 53(8), 933-957.
          https://doi.org/10.1080/02786826.2019.1614522
        - Gopalakrishnan, R., & Hogan, C. J. (2012). "Coulomb-influenced
          collisions in aerosols and dusty plasmas." Physical Review E, 85(2).
          https://doi.org/10.1103/PhysRevE.85.026410
    """
    # Calculate the pairwise sum of radii
    sum_of_radii: Union[float, NDArray[np.float64]]
    if isinstance(particle_radius, np.ndarray):
        sum_of_radii = particle_radius[:, np.newaxis] + particle_radius
    else:
        sum_of_radii = 2.0 * particle_radius

    # Calculate reduced mass
    reduced_mass: Union[float, NDArray[np.float64]]
    if isinstance(particle_mass, np.ndarray):
        reduced_mass = get_reduced_self_broadcast(particle_mass)
    else:
        reduced_mass = 0.5 * particle_mass

    # Calculate reduced friction factor
    reduced_friction_factor: Union[float, NDArray[np.float64]]
    if isinstance(friction_factor, np.ndarray):
        reduced_friction_factor = get_reduced_self_broadcast(friction_factor)
    else:
        reduced_friction_factor = 0.5 * friction_factor

    # Calculate the kinetic and continuum enhancements
    kinetic_enhance = coulomb_enhancement.get_coulomb_kinetic_limit(
        coulomb_potential_ratio
    )
    continuum_enhance = coulomb_enhancement.get_coulomb_continuum_limit(
        coulomb_potential_ratio
    )

    # Final calculation of diffusive Knudsen number
    numerator = (
        np.sqrt(temperature * BOLTZMANN_CONSTANT * reduced_mass)
        / reduced_friction_factor
    )
    denominator = sum_of_radii * continuum_enhance / kinetic_enhance

    return numerator / denominator
