"""Particle diffusion coefficient calculation."""

from typing import Union

import numpy as np
from numpy.typing import NDArray

from particula.gas.properties.dynamic_viscosity import get_dynamic_viscosity
from particula.gas.properties.mean_free_path import get_molecule_mean_free_path
from particula.particles.properties.aerodynamic_mobility_module import (
    get_aerodynamic_mobility,
)
from particula.particles.properties.knudsen_number_module import (
    get_knudsen_number,
)
from particula.particles.properties.slip_correction_module import (
    get_cunningham_slip_correction,
)
from particula.util.constants import BOLTZMANN_CONSTANT
from particula.util.validate_inputs import validate_inputs


@validate_inputs(
    {
        "temperature": "positive",
        "aerodynamic_mobility": "nonnegative",
    }
)
def get_diffusion_coefficient(
    temperature: Union[float, NDArray[np.float64]],
    aerodynamic_mobility: Union[float, NDArray[np.float64]],
    boltzmann_constant: float = BOLTZMANN_CONSTANT,
) -> Union[float, NDArray[np.float64]]:
    """Calculate the diffusion coefficient of a particle based on temperature
    and aerodynamic mobility.

    The diffusion coefficient (D) can be computed using:

    - D = k_B T × B
        - D is the diffusion coefficient in m²/s,
        - k_B is the Boltzmann constant in J/K,
        - T is the temperature in Kelvin,
        - B is the aerodynamic mobility in m²/s.

    Arguments:
        - temperature : Temperature in Kelvin (K).
        - aerodynamic_mobility : Aerodynamic mobility in m²/s.
        - boltzmann_constant : Boltzmann constant in J/K.

    Returns:
        - The diffusion coefficient of the particle in m²/s.

    Examples:
        ``` py title="Example"
        import particula as par
        par.particles.get_diffusion_coefficient(
            temperature=300.0, aerodynamic_mobility=1.0e-8
        )
        # Output: ...
        ```

    References:
        - Einstein, A. (1905). "On the movement of small particles suspended
          in stationary liquids required by the molecular-kinetic theory of
          heat." Annalen der Physik, 17(8), 549–560.
        - "Stokes-Einstein equation," Wikipedia,
          https://en.wikipedia.org/wiki/Stokes%E2%80%93Einstein_equation
    """
    return boltzmann_constant * temperature * aerodynamic_mobility


def get_diffusion_coefficient_via_system_state(
    particle_radius: Union[float, NDArray[np.float64]],
    temperature: float,
    pressure: float,
) -> Union[float, NDArray[np.float64]]:
    """Calculate the diffusion coefficient from system state parameters.

    This function determines the diffusion coefficient (D) of a particle by:
    1. Computing gas properties (dynamic viscosity, mean free path),
    2. Determining particle slip correction and aerodynamic mobility,
    3. Calling get_diffusion_coefficient() to get D.

    Arguments:
        - particle_radius : Particle radius in meters (m).
        - temperature : System temperature in Kelvin (K).
        - pressure : System pressure in Pascals (Pa).

    Returns:
        - The diffusion coefficient of the particle in m²/s.

    Examples:
        ``` py title="Example"
        import particula as par
        par.particles.get_diffusion_coefficient_via_system_state(
            particle_radius=1.0e-7,
            temperature=298.15,
            pressure=101325
        )
        # Output: ...
        ```

    References:
        - Millikan, R. A. (1923). "On the elementary electrical charge and the
          Avogadro constant." Physical Review, 2(2), 109–143. [check]
        - "Mass Diffusion," Wikipedia,
          https://en.wikipedia.org/wiki/Diffusion#Mass_diffusion
    """
    # Step 1: Calculate gas properties
    _dynamic_viscosity = get_dynamic_viscosity(temperature=temperature)
    _mean_free_path = get_molecule_mean_free_path(
        temperature=temperature,
        pressure=pressure,
        dynamic_viscosity=_dynamic_viscosity,
    )

    # Step 2: Particle properties in fluid
    _knudsen_number = get_knudsen_number(
        mean_free_path=_mean_free_path, particle_radius=particle_radius
    )
    _slip_correction_factor = get_cunningham_slip_correction(
        knudsen_number=_knudsen_number,
    )
    _aerodynamic_mobility = get_aerodynamic_mobility(
        particle_radius=particle_radius,
        slip_correction_factor=_slip_correction_factor,
        dynamic_viscosity=_dynamic_viscosity,
    )

    # Step 3: Calculate the particle diffusion coefficient
    return get_diffusion_coefficient(
        temperature=temperature,
        aerodynamic_mobility=_aerodynamic_mobility,
    )
