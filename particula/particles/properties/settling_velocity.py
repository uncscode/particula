"""Particle settling velocity in a fluid."""

from typing import Union

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import fminbound

from particula.util.constants import STANDARD_GRAVITY
from particula.util.validate_inputs import validate_inputs
from particula.gas.properties.dynamic_viscosity import (
    get_dynamic_viscosity,
)
from particula.gas.properties.mean_free_path import (
    get_molecule_mean_free_path,
)
from particula.gas.properties.kinematic_viscosity import (
    get_kinematic_viscosity,
)
from particula.particles.properties.slip_correction_module import (
    cunningham_slip_correction,
)
from particula.particles.properties.knudsen_number_module import (
    calculate_knudsen_number,
)
from particula.particles.properties.reynolds_number import (
    get_particle_reynolds_number,
)


@validate_inputs(
    {
        "particle_radius": "positive",
        "particle_density": "positive",
        "slip_correction_factor": "positive",
        "dynamic_viscosity": "nonnegative",
    }
)
def particle_settling_velocity(
    particle_radius: Union[float, NDArray[np.float64]],
    particle_density: Union[float, NDArray[np.float64]],
    slip_correction_factor: Union[float, NDArray[np.float64]],
    dynamic_viscosity: float,
    gravitational_acceleration: float = STANDARD_GRAVITY,
    fluid_density: float = 0.0,
) -> Union[float, NDArray[np.float64]]:
    # pylint: disable=too-many-arguments, too-many-positional-arguments
    """
    Calculate the settling velocity of a particle in a fluid.

    The settling velocity (vₛ) is given by:

        vₛ = (2 × r² × (ρₚ − ρ_f) × g × C_c) ⁄ (9 × μ)

        - vₛ : Settling velocity [m/s]
        - r : Particle radius [m]
        - ρₚ : Particle density [kg/m³]
        - ρ_f : Fluid density [kg/m³]
        - g : Gravitational acceleration [m/s²]
        - C_c : Cunningham slip correction factor [-]
        - μ : Dynamic viscosity of the fluid [Pa·s]

    Parameters:
        - particle_radius : The radius of the particle [m].
        - particle_density : The density of the particle [kg/m³].
        - slip_correction_factor : The slip correction factor to
            account for non-continuum effects [dimensionless].
        - dynamic_viscosity : The dynamic viscosity of the fluid [Pa·s].
        - gravitational_acceleration : The gravitational acceleration [m/s²].
            Defaults to standard gravity (9.80665 m/s²).

    Returns:
    --------
        - The settling velocity of the particle in the fluid [m/s].

    Example:
    --------
        ``` py title="Array input"
        particle_setting_velocity(
            particle_radius=np.array([1e-6, 1e-5, 1e-4]),
            particle_density=np.array([1000, 2000, 3000]),
            slip_correction_factor=np.array([1, 1, 1]),
            dynamic_viscosity=0.001,
        )
        ```

    References:
    ----------
        - When fluid denisty is much less than particle density, then the
            fluid density can be neglected.
        - https://en.wikipedia.org/wiki/Stokes%27_law
    """

    # Calculate the settling velocity using the given formula
    settling_velocity = (
        (2 * particle_radius) ** 2
        * (particle_density - fluid_density)
        * slip_correction_factor
        * gravitational_acceleration
        / (18 * dynamic_viscosity)
    )

    return settling_velocity


@validate_inputs(
    {
        "particle_inertia_time": "positive",
        "gravitational_acceleration": "positive",
        "slip_correction_factor": "positive",
    }
)
def get_particle_settling_velocity_via_inertia(
    particle_inertia_time: Union[float, NDArray[np.float64]],
    particle_radius: Union[float, NDArray[np.float64]],
    relative_velocity: Union[float, NDArray[np.float64]],
    slip_correction_factor: Union[float, NDArray[np.float64]],
    gravitational_acceleration: float,
    kinematic_viscosity: float,
) -> Union[float, NDArray[np.float64]]:
    # pylint: disable=too-many-arguments, too-many-positional-arguments
    """
    Calculate the gravitational settling velocity from the particle inertia time.

    The settling velocity (vₛ) is given by:

        vₛ = (g × τₚ × C_c) ⁄ f(Reₚ)

    - vₛ : Settling velocity [m/s]
    - g : Gravitational acceleration [m/s²]
    - τₚ : Particle inertia time [s]
    - C_c : Cunningham slip correction factor [-]
    - f(Reₚ) : Drag correction factor [-]

    The drag correction factor f(Reₚ) is given by:

        f(Reₚ) = 1 + 0.15 × Reₚ⁰·⁶⁸⁷

    - r : Particle radius [m]
    - v : Relative velocity between particle and fluid [m/s]
    - ν : Kinematic viscosity of the fluid [m²/s]
    - Reₚ : Particle Reynolds number [-]

    Parameters:
        - particle_inertia_time : Particle inertia time [s].
        - particle_radius : Particle radius [m].
        - relative_velocity : Relative velocity between particle and
            fluid [m/s].
        - slip_correction_factor : Cunningham slip correction factor [-].
        - gravitational_acceleration : Gravitational acceleration [m/s²].
        - kinematic_viscosity : Kinematic viscosity of the fluid [m²/s].

    Returns:
        - Particle settling velocity [m/s].

    References:
        - Ayala, O., Rosa, B., Wang, L. P., & Grabowski, W. W. (2008). Effects
          of turbulence on the geometric collision rate of sedimenting droplets.
          Part 1. Results from direct numerical simulation. New Journal of
          Physics, 10. https://doi.org/10.1088/1367-2630/10/7/075015
        - Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on
          the geometric collision rate of sedimenting droplets. Part 2. Theory
          and parameterization. New Journal of Physics, 10.
          https://doi.org/10.1088/1367-2630/10/7/075016
    """

    re_p = get_particle_reynolds_number(
        particle_radius=particle_radius,
        particle_velocity=relative_velocity,
        kinematic_viscosity=kinematic_viscosity,
    )
    drag_correction = 1 + 0.15 * re_p**0.687
    return (
        gravitational_acceleration
        * particle_inertia_time
        * slip_correction_factor
        / drag_correction
    )


def particle_settling_velocity_via_system_state(
    particle_radius: Union[float, NDArray[np.float64]],
    particle_density: Union[float, NDArray[np.float64]],
    temperature: float,
    pressure: float,
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate the settling velocity of a particle.

    Parameters:
        - particle_radius: The radius of the particle [m].
        - particle_density: The density of the particle [kg/m³].
        - temperature: The temperature of the system [K].
        - pressure: The pressure of the system [Pa].

    Returns:
        - The settling velocity of the particle [m/s].
    """

    # Step 1: Calculate the dynamic viscosity of the gas
    dynamic_viscosity = get_dynamic_viscosity(temperature=temperature)

    # Step 2: Calculate the mean free path of the gas molecules
    mean_free_path = get_molecule_mean_free_path(
        temperature=temperature,
        pressure=pressure,
        dynamic_viscosity=dynamic_viscosity,
    )

    # Step 3: Calculate the Knudsen number (characterizes flow regime)
    knudsen_number = calculate_knudsen_number(
        mean_free_path=mean_free_path, particle_radius=particle_radius
    )

    # Step 4: Calculate the slip correction factor (Cunningham correction)
    slip_correction_factor = cunningham_slip_correction(
        knudsen_number=knudsen_number,
    )

    # Step 5: Calculate the particle settling velocity
    return particle_settling_velocity(
        particle_radius=particle_radius,
        particle_density=particle_density,
        slip_correction_factor=slip_correction_factor,
        dynamic_viscosity=dynamic_viscosity,
    )


@validate_inputs(
    {
        "particle_radius": "positive",
        "particle_density": "positive",
        "fluid_density": "positive",
        "dynamic_viscosity": "nonnegative",
    }
)
def get_particle_settling_velocity_with_drag(
    particle_radius: Union[float, NDArray[np.float64]],
    particle_density: Union[float, NDArray[np.float64]],
    fluid_density: float,
    dynamic_viscosity: float,
    slip_correction_factor: Union[float, NDArray[np.float64]],
    gravitational_acceleration: float = STANDARD_GRAVITY,
    re_threshold: float = 0.1,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> Union[float, NDArray[np.float64]]:
    # pylint: disable=too-many-arguments, too-many-locals, too-many-positional-arguments
    """
    Calculate the terminal settling velocity of particle(s) in a fluid.

    Handles both small-particle (Stokes law with slip correction) and
    larger-droplet (nonlinear drag) regimes. For Re < `re_threshold`, returns
    the Stokes velocity directly; otherwise, uses a force-balance approach with
    `fminbound`. The limiting cases and equations are:

    - **Stokes regime (Re < 0.1):**

        vₛ = (2⁄9) × [ r² × (ρₚ − ρ_f) × g × C_c ] ⁄ μ

    - **Nonlinear drag regime (Re > 0.1):**

        vₛ = √[ (8 × r × (ρₚ − ρ_f) × g) ⁄ (3 × ρ_f × c_d) ]

    Where:

    - vₛ : Terminal settling velocity [m/s]
    - r : Particle radius [m]
    - ρₚ : Particle density [kg/m³]
    - ρ_f : Fluid density [kg/m³]
    - g : Gravitational acceleration [m/s²]
    - C_c : Cunningham slip correction factor [-]
    - μ : Dynamic viscosity of the fluid [Pa·s]
    - c_d : Drag coefficient [-]
    - Re : Particle Reynolds number [-]
        - Re = (2 × r × vₛ × ρ_f) ⁄ μ


    Parameters:
        - particle_radius : Particle radius [m].
        - particle_density : Particle density [kg/m³].
        - fluid_density : Fluid density [kg/m³].
        - dynamic_viscosity : Dynamic viscosity of the fluid [Pa·s].
        - slip_correction_factor : Cunningham slip correction factor [-].
        - gravitational_acceleration : Gravitational acceleration [m/s²].
            Defaults to standard gravity (9.80665 m/s²).
        - re_threshold : Reynolds-number threshold to decide which approach to
            use. Default is 0.1.
        - tol : Tolerance for the numeric solver `fminbound`. Default is 1e-6.
        - max_iter : Maximum function evaluations for `fminbound`.
            Default is 100.

    Returns:
        - Terminal settling velocity [m/s] for each particle.
    """

    # --- Step 1: Broadcast inputs to matching shapes if arrays are passed. ---
    (particle_radius_arr, particle_density_arr, slip_corr_arr) = (
        np.broadcast_arrays(
            particle_radius, particle_density, slip_correction_factor
        )
    )

    # Prepare output array (same shape as the broadcast arrays).
    velocities = np.zeros_like(particle_radius_arr, dtype=float)

    # --- Main loop: handle each element individually. ---
    it = np.nditer(
        [particle_radius_arr, particle_density_arr, slip_corr_arr],
        flags=["multi_index"],
    )
    while not it.finished:
        radius = it[0].item()
        rho_p = it[1].item()
        ccf = it[2].item()
        idx = it.multi_index

        # Step 2: Compute the Stokes velocity guess (with slip correction).
        v_stokes = particle_settling_velocity(
            particle_radius=radius,
            particle_density=rho_p,
            slip_correction_factor=ccf,
            dynamic_viscosity=dynamic_viscosity,
            gravitational_acceleration=gravitational_acceleration,
            fluid_density=fluid_density,
        )

        # -- Step 3: Check the Reynolds number for that Stokes guess. --
        kinematic_viscosity = get_kinematic_viscosity(
            dynamic_viscosity=dynamic_viscosity, fluid_density=fluid_density
        )
        re_stokes = get_particle_reynolds_number(
            particle_radius=radius,
            particle_velocity=v_stokes,
            kinematic_viscosity=kinematic_viscosity,
        )

        # If purely in the Stokes regime (re < re_threshold), use v_stokes.
        # (No need for a numeric solver.)
        if abs(re_stokes) < re_threshold:
            velocities[idx] = v_stokes
            it.iternext()
            continue

        # -- Step 4: Otherwise solve for velocity using fminbound. --
        # Form a bracket for velocity. We use the magnitude of v_stokes
        # to guess an lower and upper bound for the numeric solver.
        v_upper = max(abs(v_stokes) / 10, abs(v_stokes))

        # Minimize mismatch in [0, v_upper]
        v_solution = fminbound(
            _velocity_mismatch,
            0.0,
            v_upper,
            args=(
                radius,
                rho_p,
                fluid_density,
                kinematic_viscosity,
                gravitational_acceleration,
            ),
            xtol=tol,
            maxfun=max_iter,
        )
        velocities[idx] = v_solution

        it.iternext()

    # If inputs were scalars, return a scalar. Else return the array.
    if velocities.size == 1:
        return float(velocities)
    return velocities


def _drag_coefficient(reynolds_number: float) -> float:
    """
    Return drag coefficient c_d given a Reynolds number Re.

    Parameters:
        - reynolds_number : Reynolds number [-].

    Returns:
        - Drag coefficient c_d [-].
    """
    if reynolds_number < 1.0:
        # Guard against re = 0 => use a large number for drag_coefficient
        return 24.0 / reynolds_number if reynolds_number > 0 else np.inf
    if reynolds_number < 1000.0:
        return (24.0 / reynolds_number) * (
            1.0 + 0.15 * (reynolds_number**0.687)
        )
    return 0.44


# pylint: disable=too-many-arguments, too-many-positional-arguments
def _velocity_mismatch(
    velocity: float,
    radius: float,
    rho_p: float,
    fluid_density: float,
    kinematic_viscosity: float,
    gravitational_acceleration: float,
) -> float:
    """
    Calculate the mismatch between predicted and actual velocities.

    Parameters:
        - velocity : Current estimate of particle velocity [m/s].
        - radius : Particle radius [m].
        - rho_p : Particle density [kg/m³].
        - fluid_density : Fluid density [kg/m³].
        - dynamic_viscosity : Dynamic viscosity of the fluid [Pa·s].
        - gravitational_acceleration : Gravitational acceleration [m/s²].

    Returns:
        - Squared difference between predicted and actual velocities.
    """
    # Compute Reynolds number at velocity v
    reynolds = get_particle_reynolds_number(
        particle_radius=radius,
        particle_velocity=velocity,
        kinematic_viscosity=kinematic_viscosity,
    )
    c_d = _drag_coefficient(reynolds_number=reynolds)
    # Predicted velocity from force balance:
    velocity_pred = np.sqrt(
        (8.0 * radius * (rho_p - fluid_density) * gravitational_acceleration)
        / (3.0 * fluid_density * c_d)
    )
    return (velocity_pred - velocity) ** 2
