"""Caculate Mie angular scattering properties for aerosol particles."""
# pyright: reportReturnType=false, reportAssignmentType=false
# pyright: reportIndexIssue=false
# pyright: reportArgumentType=false, reportOperatorIssue=false
# pylint: disable=too-many-arguments, too-many-locals


from typing import Union, Tuple
from functools import lru_cache
import numpy as np
import PyMieScatt as ps


@lru_cache(maxsize=100000)
def discretize_scattering_angles(
    m_sphere: Union[complex, float],
    wavelength: float,
    diameter: Union[float, np.float64],
    min_angle: int = 0,
    max_angle: int = 180,
    angular_resolution: float = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Discretizes and caches the scattering function for a spherical particle
    with specified material properties and size. This function aims to optimize
    the performance of scattering calculations by caching results for
    frequently used parameters, reducing the need for repeated calculations.

    Parameters
    ----------
    m_sphere : Union[complex, float]
        The complex or real refractive index of the particle.
    wavelength : float
        The wavelength of the incident light in nanometers (nm).
    diameter : Union[float, np.float64]
        The diameter of the particle in nanometers (nm).
    min_angle : float, optional
        The minimum scattering angle in degrees to be considered in the
        calculation. Defaults to 0.
    max_angle : float, optional
        The maximum scattering angle in degrees to be considered in the
        calculation. Defaults to 180.
    angular_resolution : float, optional
        The resolution in degrees between calculated scattering angles.
        Defaults to 1.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    A tuple containing numpy arrays for the scattering function measurements:
        - measure: The scattering intensity as a function of angle.
        - parallel: The scattering intensity for parallel polarization.
        - perpendicular: The scattering intensity for
            perpendicular polarization.
        - unpolarized: The unpolarized scattering intensity.
    """
    measure, parallel, perpendicular, unpolarized = ps.ScatteringFunction(
        m=m_sphere,
        wavelength=wavelength,
        diameter=diameter,
        minAngle=min_angle,
        maxAngle=max_angle,
        angularResolution=angular_resolution
    )
    return measure, parallel, perpendicular, unpolarized


def calculate_scattering_angles(
    z_position: Union[float, np.float64],
    integrate_sphere_diameter_cm: float,
    tube_diameter_cm: float
) -> Tuple[float, float]:
    """
    Calculates forward and backward scattering angles for a given position
    along the z-axis within the CAPS instrument geometry.

    Parameters
    ----------
    z_position : Union[float, np.float64]
        The position along the z-axis (in cm).
    integrate_sphere_diameter_cm : float
        The diameter of the integrating sphere (in cm).
    tube_diameter_cm : float
        The diameter of the sample tube (in cm).

    Returns
    -------
    Tuple[float, float]
        A tuple containing the forward (alpha) and backward (beta)
        scattering angles in radians.
    """
    sphere_radius_cm = integrate_sphere_diameter_cm / 2
    tube_radius_cm = tube_diameter_cm / 2

    # Calculate forward scattering angle alpha
    if z_position != sphere_radius_cm:
        alpha = np.arctan(tube_radius_cm / abs(sphere_radius_cm - z_position))
    else:
        alpha = np.pi / 2  # Edge case when directly at the edge of the sphere

    # Calculate backward scattering angle beta
    if z_position != -sphere_radius_cm:
        beta = np.arctan(tube_radius_cm / abs(sphere_radius_cm + z_position))
    else:
        beta = np.pi / 2  # Edge case when directly at the other edge of sphere

    return alpha, beta


def assign_scattering_thetas(
    alpha: float,
    beta: float,
    q_mie: float,
    z_position: Union[float, np.float64],
    integrate_sphere_diameter_cm: float
) -> Tuple[float, float, float]:
    """
    Assigns scattering angles and efficiencies based on the z-axis position
    within the CAPS instrument.

    Parameters
    ----------
    alpha : float
        Forward scattering angle in radians.
    beta : float
        Backward scattering angle in radians.
    q_mie : float
        The Mie scattering efficiency.
    z_position : Union[float, np.float64]
        The position along the z-axis (in cm).
    integrate_sphere_diameter_cm : float
        The diameter of the integrating sphere (in cm).

    Returns
    -------
    Tuple[float, float, float]
        A tuple containing the forward scattering angle (theta1), backward
        scattering angle (theta2), and the ideal scattering efficiency
        (qsca_ideal) for the given z-axis position.
    """
    sphere_radius_cm = integrate_sphere_diameter_cm / 2

    # Determine the location of z_position relative to the sphere's center
    # Outside the sphere
    if z_position < -sphere_radius_cm or z_position > sphere_radius_cm:
        qsca_ideal = 0
    else:  # Inside the sphere
        qsca_ideal = q_mie

    # Calculate the effective scattering angles based on z_position
    theta1 = alpha if z_position <= sphere_radius_cm else np.pi - alpha
    theta2 = np.pi - beta if z_position >= -sphere_radius_cm else beta

    return theta1, theta2, qsca_ideal
