"""Functions to convert mass concentrations to other concentration units."""

from numpy.typing import NDArray
import numpy as np


def to_mole_fraction(
    mass_concentrations: NDArray[np.float64], molar_masses: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Convert mass concentrations to mole fractions for N components.

    Args:
    -----------
    - mass_concentrations: A list or ndarray of mass concentrations
    (SI, kg/m^3).
    - molar_masses: A list or ndarray of molecular weights (SI, kg/mol).

    Returns:
    --------
    - An ndarray of mole fractions.

    Reference:
    ----------
    The mole fraction of a component is given by the ratio of its molar
    concentration to the total molar concentration of all components.
    - https://en.wikipedia.org/wiki/Mole_fraction
    """
    # check for negative values
    if np.any(mass_concentrations < 0):
        raise ValueError("Mass concentrations must be positive")
    if np.any(molar_masses <= 0):
        raise ValueError("Molar masses must be non-zero, positive")
    # Convert mass concentrations to moles for each component
    moles = mass_concentrations / molar_masses
    # Calculate total moles in the mixture
    total_moles = np.sum(moles)
    return moles / total_moles


def to_volume_fraction(
    mass_concentrations: NDArray[np.float64], densities: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Convert mass concentrations to volume fractions for N components.

    Args:
    -----------
    - mass_concentrations: A list or ndarray of mass concentrations
    (SI, kg/m^3).
    - densities: A list or ndarray of densities of each component
    (SI, kg/m^3).

    Returns:
    --------
    - An ndarray of volume fractions.

    Reference:
    ----------
    The volume fraction of a component is calculated by dividing the volume
    of that component (derived from mass concentration and density) by the
    total volume of all components.
    - https://en.wikipedia.org/wiki/Volume_fraction
    """
    # check for negative values
    if np.any(mass_concentrations < 0):
        raise ValueError("Mass concentrations must be positive")
    if np.any(densities <= 0):
        raise ValueError("Densities must be Non-zero positive")
    # Calculate volumes for each component using mass concentration and density
    volumes = mass_concentrations / densities
    # Calculate total volume of the mixture
    total_volume = np.sum(volumes)
    # Calculate volume fractions by dividing the volume of each component by
    # the total volume
    return volumes / total_volume
