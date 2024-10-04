"""Functions to convert mass concentrations to other concentration units."""

from numpy.typing import NDArray
import numpy as np


def to_mole_fraction(
    mass_concentrations: NDArray[np.float64], molar_masses: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Convert mass concentrations to mole fractions for N components.

    If the input mass_concentrations is 1D, the summation is performed over the
    entire array. If mass_concentrations is 2D, the summation is done row-wise.

    Args:
        mass_concentrations: A list or ndarray of mass concentrations
            (SI, kg/m^3).
        molar_masses: A list or ndarray of molecular weights (SI, kg/mol).

    Returns:
        An ndarray of mole fractions.

    Reference:
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

    # Handle 1D and 2D cases separately
    if moles.ndim == 1:
        # For 1D input, sum over the entire array
        total_moles = np.sum(moles)
    elif moles.ndim == 2:
        # For 2D input, sum row-wise
        total_moles = np.sum(moles, axis=1, keepdims=True)
    else:
        raise ValueError("mass_concentrations must be either 1D or 2D")

    # Return mole fractions by dividing moles by the total moles
    return moles / total_moles


def to_volume_fraction(
    mass_concentrations: NDArray[np.float64], densities: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Convert mass concentrations to volume fractions for N components.

    If inputs are the one dimensional or float, the summation is done over the
    the whole array. It mass_concentration is a 2D array, the summation is done
    row-wise.

    Args:
        mass_concentrations: A list or ndarray of mass concentrations
            (SI, kg/m^3).
        densities: A list or ndarray of densities of each component
            (SI, kg/m^3).

    Returns:
        An ndarray of volume fractions.

    Reference:
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
    # Check if the input is 1D or 2D
    if volumes.ndim == 1:
        # For 1D input, sum over the entire array
        total_volume = np.sum(volumes)
    elif volumes.ndim == 2:
        # For 2D input, sum row-wise
        total_volume = np.sum(volumes, axis=1, keepdims=True)
    else:
        raise ValueError("mass_concentrations must be either 1D or 2D")
    # Calculate volume fractions by dividing the volume of each component by
    # the total volume
    return volumes / total_volume


def to_mass_fraction(
    mass_concentrations: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Convert mass concentrations to mass fractions for N components.

    If inputs are one-dimensional or float, the summation is done over the
    entire array. If mass_concentration is a 2D array, the summation is done
    row-wise.

    Args:
        mass_concentrations: A list or ndarray of mass concentrations
            (SI, kg/m^3).

    Returns:
        An ndarray of mass fractions.

    Reference:
        The mass fraction of a component is calculated by dividing the mass
        concentration of that component by the total mass concentration of
        all components.
        - https://en.wikipedia.org/wiki/Mass_fraction_(chemistry)
    """
    # check for negative values
    if np.any(mass_concentrations < 0):
        raise ValueError("Mass concentrations must be positive")

    # Calculate total mass of the mixture
    # Check if the input is 1D or 2D
    if mass_concentrations.ndim == 1:
        # For 1D input, sum over the entire array
        total_mass = np.sum(mass_concentrations)
    elif mass_concentrations.ndim == 2:
        # For 2D input, sum row-wise
        total_mass = np.sum(mass_concentrations, axis=1, keepdims=True)
    else:
        raise ValueError("mass_concentrations must be either 1D or 2D")

    # Calculate mass fractions by dividing each component by the total mass
    return mass_concentrations / total_mass
