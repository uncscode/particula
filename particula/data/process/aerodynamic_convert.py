"""
Convert between aerodynamic and physical radii of particles.
"""

from typing import Union
from functools import partial
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import fsolve

from particula.next.particles.properties import (
    calculate_knudsen_number,
    cunningham_slip_correction,
    particle_aerodynamic_length,
)
from particula.next.gas.properties.mean_free_path import (
    molecule_mean_free_path,
)


def _cost_aerodynamic_radius(
    guess_aerodynamic_radius: Union[float, NDArray[np.float64]],
    mean_free_path_air: Union[float, NDArray[np.float64]],
    particle_radius: Union[float, NDArray[np.float64]],
    **kwargs,
) -> Union[float, NDArray[np.float64]]:
    """
    Optimization cost function to determine the aerodynamic radius of a
    particle.

    Arguments:
        guess_aerodynamic_radius: The initial guess for the aerodynamic radius.
        mean_free_path_air: The mean free path of air molecules.
        particle_radius: The known physical radius of the particle.
        kwargs: Additional keyword arguments for the optimization.
            - density (float): The density of the particle. Default is
                1500 kg/m^3.
            - reference_density (float): The reference density for the
                aerodynamic radius calculation. Default is 1000 kg/m^3.
            - aerodynamic_shape_factor (float): The aerodynamic shape factor.
                Default is 1.0.

    Returns:
        The squared error between the guessed aerodynamic radius and
            the calculated aerodynamic radius.
    """
    # Calculate the physical Knudsen number and corresponding
    # slip correction factor
    physical_knudsen_number = calculate_knudsen_number(
        mean_free_path_air, particle_radius
    )
    physical_slip_correction = cunningham_slip_correction(
        knudsen_number=physical_knudsen_number
    )

    # Calculate the Knudsen number and slip correction for the guessed
    # aerodynamic radius
    guess_aerodynamic_knudsen_number = calculate_knudsen_number(
        mean_free_path_air, guess_aerodynamic_radius
    )
    guess_aerodynamic_slip_correction = cunningham_slip_correction(
        knudsen_number=guess_aerodynamic_knudsen_number
    )

    # Calculate the aerodynamic radius based on the guessed parameters
    new_aerodynamic_radius = particle_aerodynamic_length(
        physical_length=particle_radius,
        physical_slip_correction_factor=physical_slip_correction,
        aerodynamic_slip_correction_factor=guess_aerodynamic_slip_correction,
        density=kwargs.get("density", 1500),
        reference_density=kwargs.get("reference_density", 1000),
        aerodynamic_shape_factor=kwargs.get("aerodynamic_shape_factor", 1.0),
    )

    # Return the squared error between the guessed and calculated
    # aerodynamic radius
    return (new_aerodynamic_radius - guess_aerodynamic_radius) ** 2


def _cost_physical_radius(
    guess_physical_radius: Union[float, NDArray[np.float64]],
    mean_free_path_air: Union[float, NDArray[np.float64]],
    aerodynamic_radius: Union[float, NDArray[np.float64]],
    **kwargs,
) -> Union[float, NDArray[np.float64]]:
    """
    Optimization cost function to determine the physical radius of a particle.

    Arguments:
        guess_physical_radius: The initial guess for the physical radius.
        mean_free_path_air: The mean free path of air molecules.
        aerodynamic_radius: The known aerodynamic radius of the particle.
        kwargs: Additional keyword arguments for the optimization
            - density (float): The density of the particle. Default is
                1500 kg/m^3.
            - reference_density (float): The reference density for the
                aerodynamic radius calculation. Default is 1000 kg/m^3.
            - aerodynamic_shape_factor (float): The aerodynamic shape factor.
                Default is 1.0.

    Returns:
        The squared error between the guessed physical radius and the
        calculated aerodynamic radius.
    """
    # Calculate the physical Knudsen number and corresponding slip correction
    # factor
    guess_physical_knudsen_number = calculate_knudsen_number(
        mean_free_path_air, guess_physical_radius
    )
    guess_physical_slip_correction = cunningham_slip_correction(
        knudsen_number=guess_physical_knudsen_number
    )

    # Calculate the aerodynamic Knudsen number and corresponding slip
    # correction factor
    aerodynamic_knudsen_number = calculate_knudsen_number(
        mean_free_path_air, aerodynamic_radius
    )
    aerodynamic_slip_correction = cunningham_slip_correction(
        knudsen_number=aerodynamic_knudsen_number
    )

    # Calculate the aerodynamic radius based on the guessed physical radius
    new_aerodynamic_radius = particle_aerodynamic_length(
        physical_length=guess_physical_radius,
        physical_slip_correction_factor=guess_physical_slip_correction,
        aerodynamic_slip_correction_factor=aerodynamic_slip_correction,
        density=kwargs.get("density", 1500),
        reference_density=kwargs.get("reference_density", 1000),
        aerodynamic_shape_factor=kwargs.get("aerodynamic_shape_factor", 1.0),
    )

    # Return the squared error between the guessed and calculated aerodynamic
    # radius
    return (new_aerodynamic_radius - aerodynamic_radius) ** 2


# pylint: disable=too-many-arguments
def convert_aerodynamic_to_physical_radius(
    aerodynamic_radius: Union[float, NDArray[np.float64]],
    pressure: float,
    temperature: float,
    particle_density: float,
    aerodynamic_shape_factor: float = 1.0,
    reference_density: float = 1000.0,
) -> Union[float, NDArray[np.float64]]:
    """
    Convert aerodynamic radius to physical radius for a particle or an array
    of particles.

    Arguments:
        aerodynamic_radius: The aerodynamic radius or array of radii to be
            converted.
        pressure: The ambient pressure in Pascals.
        temperature: The ambient temperature in Kelvin.
        particle_density: The density of the particles in kg/m^3.
        aerodynamic_shape_factor: The aerodynamic shape factor. Default is 1.0.
        reference_density: The reference density for the aerodynamic radius
            in kg/m^3. Default is 1000 kg/m^3.

    Returns:
        The physical radius or array of radii corresponding to the aerodynamic
        radius/radii.
    """
    # Calculate the mean free path of air
    mean_free_path_air = molecule_mean_free_path(
        temperature=temperature, pressure=pressure
    )

    # Prepare additional parameters for optimization
    keywords = {
        "density": particle_density,
        "aerodynamic_shape_factor": aerodynamic_shape_factor,
        "reference_density": reference_density,
    }

    # Initial guess for physical radius is the same as the aerodynamic radius
    initial_physical_guess = aerodynamic_radius

    # Partially apply the cost function with the keyword arguments
    cost_function_with_kwargs = partial(
        _cost_physical_radius,
        mean_free_path_air=mean_free_path_air,
        aerodynamic_radius=aerodynamic_radius,
        **keywords
    )

    # Solve for physical radius using fsolve
    optimal_physical_radius = fsolve(
        cost_function_with_kwargs,
        initial_physical_guess
    )

    return optimal_physical_radius  # type: ignore


# pylint: disable=too-many-arguments
def convert_physical_to_aerodynamic_radius(
    physical_radius: Union[float, NDArray[np.float64]],
    pressure: float,
    temperature: float,
    particle_density: float,
    aerodynamic_shape_factor: float = 1.0,
    reference_density: float = 1000.0,
) -> Union[float, NDArray[np.float64]]:
    """
    Convert physical radius to aerodynamic radius for a particle or an array
    of particles.

    Arguments:
        physical_radius: The physical radius or array of radii to be converted.
        pressure: The ambient pressure in Pascals.
        temperature: The ambient temperature in Kelvin.
        particle_density: The density of the particles in kg/m^3.
        aerodynamic_shape_factor: The aerodynamic shape factor. Default is 1.0.
        reference_density: The reference density for the aerodynamic radius
            in kg/m^3. Default is 1000 kg/m^3.

    Returns:
        The aerodynamic radius or array of radii corresponding to the physical
        radius/radii.
    """
    # Calculate the mean free path of air
    mean_free_path_air = molecule_mean_free_path(
        temperature=temperature, pressure=pressure
    )

    # Prepare additional parameters for optimization
    keywords = {
        "density": particle_density,
        "aerodynamic_shape_factor": aerodynamic_shape_factor,
        "reference_density": reference_density,
    }

    # Initial guess for aerodynamic radius is the same as the physical radius
    initial_guess = physical_radius

    # Partially apply the cost function with the keyword arguments
    cost_function_with_kwargs = partial(
        _cost_aerodynamic_radius,
        mean_free_path_air=mean_free_path_air,
        particle_radius=physical_radius,
        **keywords
    )

    # Solve for aerodynamic radius using fsolve
    optimal_aerodynamic_radius = fsolve(
        cost_function_with_kwargs,
        initial_guess
    )

    return optimal_aerodynamic_radius  # type: ignore
