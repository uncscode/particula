"""Sort bins of distribution representation."""

from typing import Union

import numpy as np
from numpy.typing import NDArray


def get_sorted_bins_by_radius(
    radius: NDArray[np.float64],
    distribution: NDArray[np.float64],
    concentration: NDArray[np.float64],
    charge: Union[NDArray[np.float64], float],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    Union[NDArray[np.float64], float],
]:
    """Ensure distribution bins are sorted by increasing radius.

    Arguments:
        - radius : The radii of the particles.
        - distribution : The distribution of particle sizes or masses.
        - concentration : The concentration of each particle size or mass.
        - charge : (Optional) charge per particle; scalar or NumPy array.

    Returns:
        - distribution : The sorted distribution of particle sizes or masses.
        - concentration : The sorted concentration of each particle size/mass.
        - charge : The sorted charge of each particle size/mass.
    """
    order = np.argsort(radius)
    is_sorted = np.array_equal(order, np.arange(radius.size))
    if is_sorted:
        return distribution, concentration, charge

    distribution = distribution[order]
    concentration = concentration[order]

    if isinstance(charge, np.ndarray) and charge.shape == radius.shape:
        charge = charge[order]

    return distribution, concentration, charge
