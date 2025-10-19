"""Module for calculating Knudsen number."""

import logging
from typing import Union

import numpy as np
from numpy.typing import NDArray

from particula.util.validate_inputs import validate_inputs

logger = logging.getLogger("particula")


@validate_inputs(
    {
        "mean_free_path": "nonnegative",
        "particle_radius": "nonnegative",
    }
)
def get_knudsen_number(
    mean_free_path: Union[float, NDArray[np.float64]],
    particle_radius: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """Calculate Knudsen number (Kn) from mean free path and radius.

    The Knudsen number (Kn) indicates whether a flow is in the continuum
    regime or the free molecular regime. It is computed by:

    - Kn = λ / r
        - Kn is the Knudsen number (dimensionless),
        - λ is the mean free path in meters (m),
        - r is the particle radius in meters (m).

    Arguments:
        - mean_free_path : Mean free path of the gas molecules in meters (m).
        - particle_radius : Radius of the particle in meters (m).

    Returns:
        - The Knudsen number, which is the ratio of the mean free path to the
            particle radius.

    Examples:
        ``` py title="Example Usage"
        import particula as par
        par.particles.get_knudsen_number(6.5e-8, 1.0e-7)
        # Output: 0.65
        ```

    References:
        - Knudsen number, Wikipedia,
          https://en.wikipedia.org/wiki/Knudsen_number
    """
    if not isinstance(mean_free_path, (float, np.ndarray)) or not isinstance(
        particle_radius, (float, np.ndarray)
    ):
        message = "The input must be a float or a numpy array"
        logger.error(message)
        raise TypeError(message)
    if isinstance(mean_free_path, float) and isinstance(
        particle_radius, (float, np.ndarray)
    ):
        return mean_free_path / particle_radius

    if isinstance(mean_free_path, np.ndarray) and isinstance(
        particle_radius, np.ndarray
    ):
        if (
            (mean_free_path.size == particle_radius.size)
            or (mean_free_path.size == 1)
            or (particle_radius.size == 1)
        ):
            return mean_free_path / particle_radius

        # Reshape to (n, 1) and vector_b to (1, m) to broadcast (n, m)
        particle_radius = particle_radius[
            :, np.newaxis
        ]  # Adds a new axis, creating a column vector
        mean_free_path = mean_free_path[
            np.newaxis, :
        ]  # Adds a new axis, creating a row vector
        return mean_free_path / particle_radius

    message = (
        "The input arrays must have the same size"
        + " or one of them must have size 1"
    )
    logger.error(message)
    raise ValueError(message)
