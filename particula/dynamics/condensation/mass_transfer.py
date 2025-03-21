"""
Particle Vapor Equilibrium, condensation, and evaporation based on partial
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
from numpy.typing import NDArray
import numpy as np

# particula imports
from particula.util.constants import GAS_CONSTANT  # type: ignore
from particula.util.validate_inputs import validate_inputs


@validate_inputs(
    {
        "radius": "nonnegative",
    }
)
def get_first_order_mass_transport_k(
    radius: Union[float, NDArray[np.float64]],
    vapor_transition: Union[float, NDArray[np.float64]],
    diffusion_coefficient: Union[float, NDArray[np.float64]] = 2e-5,
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate the first-order mass transport coefficient per particle.

    This function computes the coefficient K that governs how fast mass is
    transported to or from a particle in a vapor. The equation is:

    - K = 4π × radius × D × X
        - K : Mass transport coefficient [m³/s].
        - radius : Particle radius [m].
        - D : Diffusion coefficient of the vapor [m²/s].
        - X : Vapor transition correction factor [unitless].

    Arguments:
        - radius : The radius of the particle [m].
        - vapor_transition : The vapor transition correction factor [unitless].
        - diffusion_coefficient : The diffusion coefficient of the vapor [m²/s].
          Defaults to 2e-5 (approx. air).

    Returns:
        - The first-order mass transport coefficient per particle [m³/s].

    Examples:
        ```py title="Float input"
        import particula as par
        par.dynamics.get_first_order_mass_transport_k(
            radius=1e-6,
            vapor_transition=0.6,
            diffusion_coefficient=2e-9
        )
        # Output: 1.5079644737231005e-14
        ```

        ```py title="Array input"
        import particula as par
        par.dynamics.get_first_order_mass_transport_k(
            radius=np.array([1e-6, 2e-6]),
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
        radius = radius[:, np.newaxis]  # type: ignore
    return 4 * np.pi * radius * diffusion_coefficient * vapor_transition  # type: ignore


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
    """
    Calculate the mass transfer rate for a particle.

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
    {"mass_rate": "finite", "radius": "nonnegative", "density": "positive"}
)
def get_radius_transfer_rate(
    mass_rate: Union[float, NDArray[np.float64]],
    radius: Union[float, NDArray[np.float64]],
    density: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """
    Convert mass rate to radius growth/evaporation rate.

    This function converts the mass transfer rate (dm/dt) into a radius
    change rate (dr/dt). The equation is:

    - dr/dt = (1 / 4πr²ρ) × dm/dt
        - dr/dt : Radius change rate [m/s].
        - r : Particle radius [m].
        - ρ : Particle density [kg/m³].
        - dm/dt : Mass change rate [kg/s].

    Arguments:
        - mass_rate : The mass transfer rate [kg/s].
        - radius : The radius of the particle [m].
        - density : The density of the particle [kg/m³].

    Returns:
        - The radius growth (or evaporation) rate [m/s].

    Examples:
        ```py title="Single value input"
        import particula as par
        par.dynamics.radius_transfer_rate(
            mass_rate=1e-21,
            radius=1e-6,
            density=1000
        )
        # Output: 7.95774715e-14
        ```

        ```py title="Array input"
        import particula as par
        par.dynamics.radius_transfer_rate(
            mass_rate=np.array([1e-21, 2e-21]),
            radius=np.array([1e-6, 2e-6]),
            density=1000
        )
        # Output: array([7.95774715e-14, 1.98943679e-14])
        ```
    """
    if isinstance(mass_rate, np.ndarray) and mass_rate.ndim == 2:
        radius = radius[:, np.newaxis]  # type: ignore
    return mass_rate / (density * 4 * np.pi * radius**2)  # type: ignore


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
    """
    Route mass transfer calculation to single or multiple-species routines.

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
    """
    Calculate mass transfer for a single gas species.

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
    # Step 1: Calculate the total mass to be transferred
    # (accounting for particle concentration)
    mass_to_change = mass_rate * time_step * particle_concentration
    # Step 2: Calculate the total requested mass
    total_requested_mass = mass_to_change.sum()
    # Step 3: Scale the mass if total requested mass exceeds available gas mass
    if total_requested_mass > gas_mass:
        scaling_factor = gas_mass / total_requested_mass
        mass_to_change *= scaling_factor
    # Step 4: Limit condensation by available gas mass
    condensible_mass_transfer = np.minimum(mass_to_change, gas_mass)
    # Step 5: Limit evaporation by available particle mass
    evaporative_mass_transfer = np.maximum(
        mass_to_change, -particle_mass * particle_concentration
    )
    # Step 6: Determine final transferable mass (condensation or evaporation)
    transferable_mass = np.where(
        mass_to_change > 0,  # Condensation scenario
        condensible_mass_transfer,  # Limited by gas mass
        evaporative_mass_transfer,  # Limited by particle mass
    )
    return transferable_mass


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
    """
    Calculate mass transfer for multiple gas species.

    Here, gas_mass has multiple elements (each species). For each species,
    this function calculates mass_to_change for all particle bins:

    - mass_to_change = mass_rate × time_step × particle_concentration

    Then it limits or scales that mass based on available gas mass and
    particle mass in each species bin.

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
    # Step 1: Calculate the total mass to change
    # (considering particle concentration)
    mass_to_change = (
        mass_rate * time_step * particle_concentration[:, np.newaxis]
    )

    # Step 2: Total requested mass for each gas species (sum over particles)
    total_requested_mass = mass_to_change.sum(axis=0)

    # Step 3: Create scaling factors where requested mass exceeds available
    # gas mass
    scaling_factors = np.ones_like(mass_to_change)
    scaling_mask = total_requested_mass > gas_mass

    # Apply scaling where needed (scaling along the gas species axis)
    scaling_factors[:, scaling_mask] = (
        gas_mass[scaling_mask] / total_requested_mass[scaling_mask]
    )

    # Step 4: Apply scaling factors to the mass_to_change
    mass_to_change *= scaling_factors

    # Step 5: Limit condensation by available gas mass
    condensible_mass_transfer = np.minimum(np.abs(mass_to_change), gas_mass)

    # Step 6: Limit evaporation by available particle mass
    evaporative_mass_transfer = np.maximum(
        mass_to_change, -particle_mass * particle_concentration[:, np.newaxis]
    )

    # Step 7: Determine the final transferable mass
    # (condensation or evaporation)
    transferable_mass = np.where(
        mass_to_change > 0,  # Condensation scenario
        condensible_mass_transfer,  # Limited by gas mass
        evaporative_mass_transfer,  # Limited by particle mass
    )

    return transferable_mass
