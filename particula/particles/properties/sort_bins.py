"""
Sort bins of distribution representation
"""
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
        - charge : The (optional) charge per particle; can be a scalar or NumPy array.

    Returns:
        - distribution : The sorted distribution of particle sizes or masses.
        - concentration : The sorted concentration of each particle size/mass.
        - charge : The sorted charge of each particle size/mass.
    """
    sort_index = np.argsort(radius)
    sorting_is_needed = not np.array_equal(sort_index, np.arange(radius.size))

    if sorting_is_needed:
        distribution = np.take(distribution, sort_index, axis=0)
        concentration = np.take(concentration, sort_index, axis=0)

        charge_is_same_shape_and_ndarray = (
            isinstance(charge, np.ndarray)
            and charge.shape == radius.shape
        )
        if charge_is_same_shape_and_ndarray:
            charge = np.take(charge, sort_index, axis=0)
    return distribution, concentration, charge
