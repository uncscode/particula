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
    molecule_mean_free_path,
)
from particula.particles.properties.slip_correction_module import (
    cunningham_slip_correction,
)
from particula.particles.properties.knudsen_number_module import (
    calculate_knudsen_number,
)


def particle_settling_velocity(
    particle_radius: Union[float, NDArray[np.float64]],
    particle_density: Union[float, NDArray[np.float64]],
    slip_correction_factor: Union[float, NDArray[np.float64]],
    dynamic_viscosity: float,
    gravitational_acceleration: float = STANDARD_GRAVITY,
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate the settling velocity of a particle in a fluid.

    Arguments:
        particle_radius: The radius of the particle [m].
        particle_density: The density of the particle [kg/m³].
        slip_correction_factor: The slip correction factor to
            account for non-continuum effects [dimensionless].
        gravitational_acceleration: The gravitational acceleration.
            Defaults to standard gravity [9.80665 m/s²].
        dynamic_viscosity: The dynamic viscosity of the fluid [Pa*s].

    Returns:
        The settling velocity of the particle in the fluid [m/s].

    """

    # Calculate the settling velocity using the given formula
    settling_velocity = (
        (2 * particle_radius) ** 2
        * particle_density
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
    # pylint disable=too-many-arguments
    """
    Calculate the gravitational settling velocity from the particle inertia
    time.

    The settling velocity (v_s) is given by:

        v_s = g * τ_p * C_c / f(Re_p)

    - v_s : Settling velocity [m/s]
    - g (gravitational_acceleration) : Gravitational acceleration [m/s²]
    - τ_p (particle_inertia_time) : Particle inertia time [s]
    - C_c (slip_correction_factor) : Cunningham slip correction factor [-]

    The drag correction factor f(Re_p) is given by:

        f(Re_p) = 1 + 0.15 Re_p^(0.687)

    where the particle Reynolds number is:

        Re_p = (2 r v) / v

    - v (relative_velocity) : Relative velocity between particle and fluid
        [m/s]

    Arguments:
    ----------
        - particle_inertia_time : Particle inertia time [s]
        - gravitational_acceleration : Gravitational acceleration [m/s²]
        - slip_correction_factor : Cunningham slip correction factor [-]

    Returns:
    --------
        - Particle settling velocity [m/s]

    References:
    -----------
        -Note miss match in references definition. Using Part 1, eq 3-4
        -Ayala, O., Rosa, B., Wang, L. P., & Grabowski, W. W. (2008). Effects
        of turbulence on the geometric collision rate of sedimenting droplets.
        Part 1. Results from direct numerical simulation. New Journal of
        Physics, 10. https://doi.org/10.1088/1367-2630/10/7/075015
        - Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on
        the geometric collision rate of sedimenting droplets. Part 2. Theory
        and parameterization. New Journal of Physics, 10.
        https://doi.org/10.1088/1367-2630/10/7/075016
    """

    re_p = (2 * particle_radius * relative_velocity) / kinematic_viscosity
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

    Arguments:
        particle_radius: The radius of the particle in meters (m).
        particle_density: The density of the particle (kg/m³).
        temperature: The temperature of the system in Kelvin (K).
        pressure: The pressure of the system in Pascals (Pa).

    Returns:
        The settling velocity of the particle in meters per second (m/s).
    """

    # Step 1: Calculate the dynamic viscosity of the gas
    dynamic_viscosity = get_dynamic_viscosity(temperature=temperature)

    # Step 2: Calculate the mean free path of the gas molecules
    mean_free_path = molecule_mean_free_path(
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


# def get_particle_settling_velocity_with_drag(
#     particle_radius: Union[float, NDArray[np.float64]],
#     particle_density: Union[float, NDArray[np.float64]],
#     fluid_density: float,
#     dynamic_viscosity: float,
#     slip_correction_factor: Union[float, NDArray[np.float64]],
#     gravitational_acceleration: float = STANDARD_GRAVITY,
#     re_threshold: float = 1.0,
#     tol: float = 1e-6,
#     max_iter: int = 100,
# ) -> float:
#     # pylint disable=too-many-arguments
#     """
#     Calculate the terminal settling velocity of a particle in a fluid,
#     handling both the small-particle (Stokes law with slip correction)
#     and large-droplet (nonlinear drag) regimes.

#     For small particles (Re < re_threshold), the velocity is computed using
#     Stokes law:

#         v = (2/9) * (particle_radius**2 * (particle_density - fluid_density) *
#                      gravitational_acceleration * slip_correction_factor)
#                      / dynamic_viscosity

#     For larger droplets (Re >= re_threshold), we solve iteratively for v from:

#         v = sqrt( (8 * particle_radius * (particle_density - fluid_density) *
#                    gravitational_acceleration) / (3 * fluid_density * C_d) )

#     where the drag coefficient C_d is defined by:
#         - If Re < 1:      C_d = 24 / Re   (with care taken if Re is 0)
#         - If 1 <= Re < 1000: C_d = (24 / Re) * (1 + 0.15 * Re**0.687)
#         - If Re >= 1000:   C_d = 0.44

#     Args:
#         particle_radius: Particle radius in meters.
#         particle_density: Particle density in kg/m³.
#         fluid_density: Fluid density in kg/m³.
#         dynamic_viscosity: Dynamic viscosity of the fluid in Pa·s.
#         slip_correction_factor: Factor to correct for noncontinuum effects
#             (used in the small-particle regime). Defaults to 1.
#         gravitational_acceleration: Gravitational acceleration (m/s²).
#         re_threshold: Reynolds number threshold to decide which formulation to
#             use. Defaults to 1.
#         tol: Tolerance for convergence in the iterative solution.
#         max_iter: Maximum number of iterations for the iterative solution.

#     Returns:
#         Terminal settling velocity in m/s.
#     """
#     # Use the Stokes law (with slip correction) as an initial estimate.
#     # (If you wish to include buoyancy for aerosols, replace particle_density
#     # with (particle_density-fluid_density))
#     v_stokes = (
#         (2 / 9)
#         * (
#             particle_radius**2
#             * (particle_density - fluid_density)
#             * gravitational_acceleration
#             * slip_correction_factor
#         )
#         / dynamic_viscosity
#     )

#     # Compute the Reynolds number based on the Stokes estimate:
#     Re_stokes = (
#         2 * particle_radius * v_stokes * fluid_density / dynamic_viscosity
#     )

#     # If the particle is small (low Re), return the Stokes law result.
#     if np.all(Re_stokes < re_threshold):
#         return v_stokes

#     # For larger particles, solve iteratively using a drag coefficient that
#     # depends on Re.
#     v = v_stokes  # initial guess
#     for _ in range(max_iter):
#         Re = 2 * particle_radius * v * fluid_density / dynamic_viscosity
#         if np.all(Re < 1):
#             C_d = 24 / Re if Re != 0 else np.inf
#         elif np.all(Re < 1000):
#             C_d = (24 / Re) * (1 + 0.15 * Re**0.687)
#         else:
#             C_d = 0.44

#         # Compute the new estimate for v from the force balance.
#         v_new = np.sqrt(
#             (
#                 8
#                 * particle_radius
#                 * (particle_density - fluid_density)
#                 * gravitational_acceleration
#             )
#             / (3 * fluid_density * C_d)
#         )

#         if np.all(abs(v_new - v) < tol):
#             return v_new
#         v = v_new

#     raise RuntimeError(
#         "Settling velocity did not converge within the max iterations"
#     )


def get_particle_settling_velocity_with_drag(
    particle_radius: Union[float, NDArray[np.float64]],
    particle_density: Union[float, NDArray[np.float64]],
    fluid_density: float,
    dynamic_viscosity: float,
    slip_correction_factor: Union[float, NDArray[np.float64]],
    gravitational_acceleration: float = STANDARD_GRAVITY,
    re_threshold: float = 1.0,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> Union[float, NDArray[np.float64]]:
    # pylint disable=too-many-arguments too-many-locals
    """
    Calculate the terminal settling velocity of particle(s) in a fluid,
    handling both small-particle (Stokes law + slip correction) and
    larger-droplet (nonlinear drag) regimes. For Re < `re_threshold`, returns
    the Stokes velocity directly; otherwise, uses a force-balance approach with
    `fminbound`.

    Parameters
    ----------
    particle_radius : float or array
        Particle radius [m].
    particle_density : float or array
        Particle density [kg/m^3].
    fluid_density : float
        Fluid density [kg/m^3].
    dynamic_viscosity : float
        Fluid dynamic viscosity [Pa·s].
    slip_correction_factor : float or array
        Cunningham slip correction factor (for small aerosol particles).
    gravitational_acceleration : float, optional
        Gravitational acceleration [m/s^2]. Default is 9.81 m/s^2.
    re_threshold : float, optional
        Reynolds-number threshold to decide which approach to use.
    tol : float, optional
        Tolerance for `fminbound`.
    max_iter : int, optional
        Maximum function evaluations for `fminbound`.

    Returns
    -------
    float or np.ndarray
        Terminal settling velocity [m/s] for each particle.
    """

    # --- Step 1: Broadcast inputs to matching shapes if arrays are passed. ---
    (particle_radius_arr, particle_density_arr, slip_corr_arr) = (
        np.broadcast_arrays(
            particle_radius, particle_density, slip_correction_factor
        )
    )

    # Prepare output array (same shape as the broadcast arrays).
    velocities = np.zeros_like(particle_radius_arr, dtype=float)

    # --- Helper function for piecewise drag coefficient. ---
    def drag_coefficient(Re: float) -> float:
        """Return drag coefficient C_d given a Reynolds number Re."""
        if Re < 1.0:
            # Guard against Re = 0 => use a large number for drag_coefficient
            return 24.0 / Re if Re > 0 else np.inf
        elif Re < 1000.0:
            return (24.0 / Re) * (1.0 + 0.15 * (Re**0.687))
        else:
            return 0.44

    # --- Main loop: handle each element individually. ---
    it = np.nditer(
        [particle_radius_arr, particle_density_arr, slip_corr_arr],
        flags=["multi_index"],
    )
    while not it.finished:
        r = it[0].item()
        rho_p = it[1].item()
        ccf = it[2].item()
        idx = it.multi_index

        # -- Step 2: Compute the Stokes velocity guess (with slip correction). --
        v_stokes = (
            (2.0 / 9.0)
            * (
                r**2
                * (rho_p - fluid_density)
                * gravitational_acceleration
                * ccf
            )
            / dynamic_viscosity
        )

        # -- Step 3: Check the Reynolds number for that Stokes guess. --
        Re_stokes = 2.0 * r * v_stokes * fluid_density / dynamic_viscosity

        # If purely in the Stokes regime (Re < re_threshold), use v_stokes.
        # (No need for a numeric solver.)
        if abs(Re_stokes) < re_threshold:
            velocities[idx] = v_stokes
            it.iternext()
            continue

        # -- Step 4: Otherwise solve for velocity using fminbound. --
        # We'll define a function that measures the mismatch in the
        # nonlinear terminal velocity equation:
        #
        #   v_expected = sqrt(
        #       (8 * r * (rho_p - fluid_density) * g) / (3 * fluid_density * C_d)
        #   )
        #
        # We want v == v_expected. So define an objective that is
        # (v_expected - v)^2 and minimize over v >= 0.

        def velocity_mismatch(v: float) -> float:
            # Compute Reynolds number at velocity v
            Re = (2.0 * r * v * fluid_density) / dynamic_viscosity
            Cd = drag_coefficient(Re)
            # Predicted velocity from force balance:
            v_pred = np.sqrt(
                (
                    8.0
                    * r
                    * (rho_p - fluid_density)
                    * gravitational_acceleration
                )
                / (3.0 * fluid_density * Cd)
            )
            return (v_pred - v) ** 2

        # Form a bracket for velocity. We use the magnitude of v_stokes
        # to guess an upper bound. If v_stokes is zero or negative, just
        # pick a small default for the bracket. Adjust as needed.
        v_upper = max(1e-10, abs(v_stokes) * 10.0)

        # Minimize mismatch in [0, v_upper]
        v_solution = fminbound(
            velocity_mismatch,
            0.0,
            v_upper,
            xtol=tol,
            maxfun=max_iter,
        )
        velocities[idx] = v_solution

        it.iternext()

    # If inputs were scalars, return a scalar. Else return the array.
    if velocities.size == 1:
        return float(velocities)
    return velocities
