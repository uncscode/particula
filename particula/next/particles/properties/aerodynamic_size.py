"""Module for aerodynamic size and shape factor of a particle in a fluid.
"""

from typing import Union
from numpy.typing import NDArray
import numpy as np


AERODYNAMIC_SHAPE_FACTOR_DICT = {
    "sphere": 1.0,
    "cube": 1.08,
    "cylinder_avg_aspect_2": 1.10,
    "cylinder_avg_aspect_5": 1.35,
    "cylinder_avg_aspect_10": 1.68,
    "spheres_cluster_3": 1.15,
    "spheres_cluster_4": 1.17,
    "bituminous_coal": 1.08,
    "quartz": 1.36,
    "sand": 1.57,
    "talc": 1.88,
}


def particle_aerodynamic_length(  # pylint: disable=too-many-positional-arguments, too-many-arguments
    physical_length: Union[float, NDArray[np.float64]],
    physical_slip_correction_factor: Union[float, NDArray[np.float64]],
    aerodynamic_slip_correction_factor: Union[float, NDArray[np.float64]],
    density: Union[float, NDArray[np.float64]],
    reference_density: float = 1000,
    aerodynamic_shape_factor: float = 1.0,
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate the aerodynamic length scale of a particle.

    The aerodynamic size is used to compare the aerodynamic properties of
    particles with their physical properties, particularly when interpreting
    aerodynamic particle sizer measurements. This is either the aerodynamic
    diameter or radius, depending on length scale input.

    Args:
        physical_length: Physical length scale of the particle (m).
        physical_slip_correction_factor: Slip correction factor for the
            particle's physical radius in the fluid (dimensionless).
        aerodynamic_slip_correction_factor: Slip correction factor for the
            particle's aerodynamic radius in the fluid (dimensionless).
        density: Density of the particle (kg/m^3).
        reference_density: Reference density for the particle, typically the
            density of water (1000 kg/m^3 by default).
        aerodynamic_shape_factor: Shape factor of the particle, accounting for
            non-sphericity (dimensionless, default is 1.0 for spherical
            particles).

    Returns:
        Aerodynamic radius of the particle (m).

    References:
        - https://en.wikipedia.org/wiki/Aerosol#Aerodynamic_diameter
        - Hinds, W.C. (1998) Aerosol Technology: Properties, behavior, and
            measurement of airborne particles. Wiley-Interscience, New York.
            pp 51-53, section 3.6.
    """
    return physical_length * np.sqrt(
        (physical_slip_correction_factor / aerodynamic_slip_correction_factor)
        * (density / (reference_density * aerodynamic_shape_factor))
    )


def get_aerodynamic_shape_factor(shape_key: str) -> float:
    """Retrieve the aerodynamic shape factor for a given particle shape.

    Args:
        shape_key: The shape of the particle as a string.

    Returns:
        The shape factor of the particle as a float.

    Raises:
        ValueError: If the shape is not found in the predefined shape
            factor dictionary.
    """
    shape_key = shape_key.strip().lower()  # Clean up the input

    # Retrieve the shape factor from the dictionary, or raise an error
    try:
        return AERODYNAMIC_SHAPE_FACTOR_DICT[shape_key]
    except KeyError as exc:
        raise ValueError(
            f"The shape factor for the shape '{shape_key}' is not available."
        ) from exc
