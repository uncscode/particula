"""Molar mass conversions with validation and array support."""

from typing import Iterable, Union

import numpy as np
from numpy.typing import NDArray

from particula.util.validate_inputs import validate_inputs

FloatArray = Union[float, NDArray[np.float64]]


@validate_inputs({"molar_mass": "positive", "other_molar_mass": "positive"})
def to_molar_mass_ratio(
    molar_mass: Union[float, Iterable[float], NDArray[np.float64]],
    other_molar_mass: float = 18.01528,
) -> FloatArray:
    """Convert to molar mass ratio (MW water / MW organic).

    Args:
        molar_mass: Molar mass of the organic compound. Must be positive.
        other_molar_mass: Reference molar mass (default water, 18.01528).
            Must be positive.

    Returns:
        Molar mass ratio. Float for scalar input; ndarray for list/array input.

    Raises:
        ValueError: If molar_mass or other_molar_mass is not positive.

    Examples:
        >>> to_molar_mass_ratio(36.03)
        0.5
        >>> to_molar_mass_ratio([36.03, 18.01528])
        array([0.5, 1. ])
        >>> to_molar_mass_ratio(np.array([72.06]))
        array([0.25])
    """
    molar_mass_array = np.asarray(molar_mass, dtype=np.float64)
    ratio = other_molar_mass / molar_mass_array
    if np.isscalar(molar_mass) and not isinstance(molar_mass, np.ndarray):
        return float(ratio)
    return ratio


@validate_inputs(
    {
        "molar_mass_ratio": "positive",
        "other_molar_mass": "positive",
    }
)
def from_molar_mass_ratio(
    molar_mass_ratio: Union[float, Iterable[float], NDArray[np.float64]],
    other_molar_mass: float = 18.01528,
) -> FloatArray:
    """Convert molar mass ratio to organic molar mass.

    Args:
        molar_mass_ratio: Molar mass ratio (MW water / MW organic).
            Must be positive.
        other_molar_mass: Reference molar mass (default water, 18.01528).
            Must be positive.

    Returns:
        Organic molar mass. Float for scalar input; ndarray for list/array
        input.

    Raises:
        ValueError: If molar_mass_ratio or other_molar_mass is not positive.

    Examples:
        >>> from_molar_mass_ratio(0.5)
        9.00764
        >>> from_molar_mass_ratio([0.5, 1.0])
        array([ 9.00764, 18.01528])
        >>> from_molar_mass_ratio(np.array([0.25]))
        array([4.50382])
    """
    molar_mass_ratio_array = np.asarray(molar_mass_ratio, dtype=np.float64)
    molar_mass = other_molar_mass * molar_mass_ratio_array
    if np.isscalar(molar_mass_ratio) and not isinstance(
        molar_mass_ratio, np.ndarray
    ):
        return float(molar_mass)
    return molar_mass
