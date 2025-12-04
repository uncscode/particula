"""Particle Vapor Equilibrium, condensation, and evaporation based on partial
pressures to calculate dm/dt or other forms of particle growth and decay.

Equation:
    - dm/dt = 4π × radius × Di × Mi × f(Kn, α) × delta_pi / (R × T)
      - radius : The particle radius in m.
      - Di : The diffusion coefficient of species i in m²/s.
      - Mi : The molar mass of species i in kg/mol.
      - f(Kn, α) : Transition function based on Knudsen number and
          accommodation coefficient.
      - delta_pi : Difference in partial pressures between gas and particle
          phases in Pa.
      - R : Gas constant in J/(mol·K).
      - T : Temperature in K.

References:
    - Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric Chemistry and
      Physics: From Air Pollution to Climate Change (3rd ed.). John
      Wiley & Sons, Inc.
    - Topping, D., & Bane, M. (2022). Introduction to Aerosol Modelling
      (D. Topping & M. Bane, Eds.). Wiley. https://doi.org/10.1002/9781119625728
    - Aerosol Modeling: Chapter 2, Equation 2.40
"""

from typing import Union

import numpy as np
from numpy.typing import NDArray

from particula.dynamics.condensation.mass_transfer_utils import (
    apply_condensation_limit,
    apply_evaporation_limit,
    apply_per_bin_limit,
    calc_mass_to_change,
)

# particula imports
from particula.util.constants import GAS_CONSTANT  # type: ignore
from particula.util.validate_inputs import validate_inputs


@validate_inputs(
    {
        "particle_radius": "nonnegative",
    }
)
def get_first_order_mass_transport_k(
    particle_radius: Union[float, NDArray[np.float64]],
    vapor_transition: Union[float, NDArray[np.float64]],
    diffusion_coefficient: Union[float, NDArray[np.float64]] = 2e-5,
) -> Union[float, NDArray[np.float64]]:
    """Calculate the first-order mass transport coefficient per particle.

    This function computes the coefficient K that governs how fast mass is
    transported to or from a particle in a vapor. The equation is:

    - K = 4π × radius × D × X
        - K : Mass transport coefficient [m³/s].
        - radius : Particle radius [m].
        - D : Diffusion coefficient of the vapor [m²/s].
        - X : Vapor transition correction factor [unitless].

    Arguments:
        - particle_radius : The radius of the particle [m].
        - vapor_transition : The vapor transition correction factor [unitless].
        - diffusion_coefficient : The diffusion coefficient of the vapor [m²/s].
          Defaults to 2e-5 (approx. air).

    Returns:
        - The first-order mass transport coefficient per particle [m³/s].

    Examples:
        ```py title="Float input"
        import particula as par
        par.dynamics.get_first_order_mass_transport_k(
            particle_radius=1e-6,
            vapor_transition=0.6,
            diffusion_coefficient=2e-9
        )
        # Output: 1.5079644737231005e-14
        ```

        ```py title="Array input"
        import particula as par
        par.dynamics.get_first_order_mass_transport_k(
            particle_radius=np.array([1e-6, 2e-6]),
            vapor_transition=np.array([0.6, 0.6]),
            diffusion_coefficient=2e-9
        )
        # Output: array([1.50796447e-14, 6.03185789e-14])
        ```

    References:
        - Aerosol Modeling: Chapter 2, Equation 2.49
        - Wikipedia contributors, "Mass diffusivity,"
          https://en.wikipedia.org/wiki/Mass_diffusivity
    """
    if (
        isinstance(vapor_transition, np.ndarray)
        and vapor_transition.dtype == np.float64
        and vapor_transition.ndim == 2
    ):  # extent radius
        particle_radius = particle_radius[:, np.newaxis]  # type: ignore
    return (
        4 * np.pi * particle_radius * diffusion_coefficient * vapor_transition
    )


@validate_inputs(
    {
        "pressure_delta": "finite",
        "first_order_mass_transport": "finite",
        "temperature": "positive",
        "molar_mass": "positive",
    }
)
def get_mass_transfer_rate(
    pressure_delta: Union[float, NDArray[np.float64]],
    first_order_mass_transport: Union[float, NDArray[np.float64]],
    temperature: Union[float, NDArray[np.float64]],
    molar_mass: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """Calculate the mass transfer rate for a particle.

    This function calculates the mass transfer rate dm/dt, leveraging the
    difference in partial pressure (pressure_delta) and the first-order
    mass transport coefficient (K). The equation is:

    - dm/dt = (K × Δp × M) / (R × T)
        - dm/dt : Mass transfer rate [kg/s].
        - K : First-order mass transport coefficient [m³/s].
        - Δp : Partial pressure difference [Pa].
        - M : Molar mass [kg/mol].
        - R : Universal gas constant [J/(mol·K)].
        - T : Temperature [K].

    Arguments:
        - pressure_delta : The difference in partial pressure [Pa].
        - first_order_mass_transport : The mass transport coefficient [m³/s].
        - temperature : The temperature [K].
        - molar_mass : The molar mass [kg/mol].

    Returns:
        - The mass transfer rate [kg/s].

    Examples:
        ```py title="Single value input"
        import particula as par
        par.dynamics.mass_transfer_rate(
            pressure_delta=10.0,
            first_order_mass_transport=1e-17,
            temperature=300.0,
            molar_mass=0.02897
        )
        # Output: 1.16143004e-21
        ```

        ```py title="Array input"
        import particula as par
        par.dynamics.mass_transfer_rate(
            pressure_delta=np.array([10.0, 15.0]),
            first_order_mass_transport=np.array([1e-17, 2e-17]),
            temperature=300.0,
            molar_mass=0.02897
        )
        # Output: array([1.16143004e-21, 3.48429013e-21])
        ```

    References:
        - Aerosol Modeling: Chapter 2, Equation 2.41
        - Seinfeld and Pandis, "Atmospheric Chemistry and Physics,"
            Equation 13.3
    """
    return np.array(
        first_order_mass_transport
        * pressure_delta
        * molar_mass
        / (GAS_CONSTANT * temperature),
        dtype=np.float64,
    )


@validate_inputs(
    {
        "mass_rate": "finite",
        "particle_radius": "nonnegative",
        "density": "positive",
    }
)
def get_radius_transfer_rate(
    mass_rate: Union[float, NDArray[np.float64]],
    particle_radius: Union[float, NDArray[np.float64]],
    density: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """Convert mass rate to radius growth/evaporation rate.

    This function converts the mass transfer rate (dm/dt) into a radius
    change rate (dr/dt). The equation is:

    - dr/dt = (1 / 4πr²ρ) × dm/dt
        - dr/dt : Radius change rate [m/s].
        - r : Particle radius [m].
        - ρ : Particle density [kg/m³].
        - dm/dt : Mass change rate [kg/s].

    Arguments:
        - mass_rate : The mass transfer rate [kg/s].
        - particle_radius : The radius of the particle [m].
        - density : The density of the particle [kg/m³].

    Returns:
        - The radius growth (or evaporation) rate [m/s].

    Examples:
        ```py title="Single value input"
        import particula as par
        par.dynamics.radius_transfer_rate(
            mass_rate=1e-21,
            particle_radius=1e-6,
            density=1000
        )
        # Output: 7.95774715e-14
        ```

        ```py title="Array input"
        import particula as par
        par.dynamics.radius_transfer_rate(
            mass_rate=np.array([1e-21, 2e-21]),
            particle_radius=np.array([1e-6, 2e-6]),
            density=1000
        )
        # Output: array([7.95774715e-14, 1.98943679e-14])
        ```
    """
    # Type narrowing: handle 2D mass_rate with array particle_radius
    radius_for_calc: Union[float, NDArray[np.float64]] = particle_radius
    if isinstance(mass_rate, np.ndarray) and mass_rate.ndim == 2:
        if isinstance(particle_radius, np.ndarray):
            radius_for_calc = particle_radius[:, np.newaxis]
    return mass_rate / (density * 4 * np.pi * radius_for_calc**2)


@validate_inputs(
    {
        "mass_rate": "finite",
        "time_step": "positive",
        "gas_mass": "nonnegative",
        "particle_mass": "nonnegative",
        "particle_concentration": "nonnegative",
    }
)
def get_mass_transfer(
    mass_rate: NDArray[np.float64],
    time_step: float,
    gas_mass: NDArray[np.float64],
    particle_mass: NDArray[np.float64],
    particle_concentration: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Route mass transfer calculation to single or multiple-species routines.

    Depending on whether gas_mass represents one or multiple species, this
    function calls either calculate_mass_transfer_single_species or
    calculate_mass_transfer_multiple_species. The primary calculation
    involves:

    - mass_to_change = mass_rate × time_step × particle_concentration

    Arguments:
        - mass_rate : The rate of mass transfer per particle [kg/s].
        - time_step : The time step for the mass transfer calculation [s].
        - gas_mass : The available mass of gas species [kg].
        - particle_mass : The mass of each particle [kg].
        - particle_concentration : The concentration of particles [#/m³].

    Returns:
        - The mass transferred (array with the same shape as particle_mass).

    Examples:
        ```py title="Single species input"
        import particula as par
        par.dynamics.get_mass_transfer(
            mass_rate=np.array([0.1, 0.5]),
            time_step=10,
            gas_mass=np.array([0.5]),
            particle_mass=np.array([1.0, 50]),
            particle_concentration=np.array([1, 0.5])
        )
        ```

        ```py title="Multiple species input"
        import particula as par
        par.dynamics.get_mass_transfer(
            mass_rate=np.array([[0.1, 0.05, 0.03], [0.2, 0.15, 0.07]]),
            time_step=10,
            gas_mass=np.array([1.0, 0.8, 0.5]),
            particle_mass=np.array([[1.0, 0.9, 0.8], [1.2, 1.0, 0.7]]),
            particle_concentration=np.array([5, 4])
        )
        ```
    """
    if gas_mass.size == 1:  # Single species case
        return get_mass_transfer_of_single_species(
            mass_rate,
            time_step,
            gas_mass,
            particle_mass,
            particle_concentration,
        )
    # Multiple species case
    return get_mass_transfer_of_multiple_species(
        mass_rate,
        time_step,
        gas_mass,
        particle_mass,
        particle_concentration,
    )


@validate_inputs(
    {
        "mass_rate": "finite",
        "time_step": "positive",
        "gas_mass": "nonnegative",
        "particle_mass": "nonnegative",
        "particle_concentration": "nonnegative",
    }
)
def get_mass_transfer_of_single_species(
    mass_rate: NDArray[np.float64],
    time_step: float,
    gas_mass: NDArray[np.float64],
    particle_mass: NDArray[np.float64],
    particle_concentration: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Calculate mass transfer for a single gas species.

    This function assumes gas_mass has a size of 1 (single species).
    It first computes the total mass_to_change per particle:

    - mass_to_change = mass_rate × time_step × particle_concentration

    Then it scales or limits that mass based on available gas mass and
    particle mass.

    Arguments:
        - mass_rate : Mass transfer rate per particle [kg/s].
        - time_step : The time step [s].
        - gas_mass : Total available mass of the gas species [kg].
        - particle_mass : The mass of each particle [kg].
        - particle_concentration : Particle concentration [#/m³].

    Returns:
        - The amount of mass transferred for the single gas species, shaped
          like particle_mass.

    Examples:
        ```py title="Single species input"
        import particula as par
        par.dynamics.get_mass_transfer_of_single_species(
            mass_rate=np.array([0.1, 0.5]),
            time_step=10,
            gas_mass=np.array([0.5]),
            particle_mass=np.array([1.0, 50]),
            particle_concentration=np.array([1, 0.5])
        )
        # Output: array([...])
        ```
    """
    mass_to_change = calc_mass_to_change(
        mass_rate, time_step, particle_concentration
    )
    mass_to_change, evap_sum, neg_mask = apply_condensation_limit(
        mass_to_change, gas_mass
    )
    mass_to_change = apply_evaporation_limit(
        mass_to_change,
        particle_mass,
        particle_concentration,
        evap_sum,
        neg_mask,
    )
    return apply_per_bin_limit(
        mass_to_change, particle_mass, particle_concentration
    )


# pylint: disable=too-many-locals
@validate_inputs(
    {
        "mass_rate": "finite",
        "time_step": "positive",
        "gas_mass": "nonnegative",
        "particle_mass": "nonnegative",
        "particle_concentration": "nonnegative",
    }
)
def get_mass_transfer_of_multiple_species(
    mass_rate: NDArray[np.float64],
    time_step: float,
    gas_mass: NDArray[np.float64],
    particle_mass: NDArray[np.float64],
    particle_concentration: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Calculate mass transfer for multiple gas species.

    Here, gas_mass has multiple elements (each species). For each species,
    this function calculates mass_to_change for all particle bins:

    - mass_to_change = mass_rate × time_step × particle_concentration

    Then it limits or scales that mass based on available gas mass and
    particle mass in each species bin.

    1. Computes the mass change each particle *would* take during `time_step`.
    2. Scales condensation so the **column sum** never exceeds `gas_mass`.
    3. Scales evaporation so the **column sum** never exceeds the particle
       inventory of that species.
    4. Clips the result so no individual bin evaporates more mass than it owns.

    Arguments:
        - mass_rate : The mass transfer rate per particle for each gas
            species [kg/s].
        - time_step : The time step [s].
        - gas_mass : The available mass of each gas species [kg].
        - particle_mass : The mass of each particle for each gas species [kg].
        - particle_concentration : The concentration of particles [#/m³].

    Returns:
        - The mass transferred for multiple gas species, matching the shape
          of (particle_mass).

    Examples:
        ```py title="Multiple species input"
        import particula as par
        par.dynamics.get_mass_transfer_of_multiple_species(
            mass_rate=np.array([[0.1, 0.05, 0.03], [0.2, 0.15, 0.07]]),
            time_step=10,
            gas_mass=np.array([1.0, 0.8, 0.5]),
            particle_mass=np.array([[1.0, 0.9, 0.8], [1.2, 1.0, 0.7]]),
            particle_concentration=np.array([5, 4])
        )
        # Output: array([...])
        ```
    """
    mass_to_change = calc_mass_to_change(
        mass_rate, time_step, particle_concentration
    )
    mass_to_change, evap_sum, neg_mask = apply_condensation_limit(
        mass_to_change, gas_mass
    )
    mass_to_change = apply_evaporation_limit(
        mass_to_change,
        particle_mass,
        particle_concentration,
        evap_sum,
        neg_mask,
    )
    return apply_per_bin_limit(
        mass_to_change, particle_mass, particle_concentration
    )
