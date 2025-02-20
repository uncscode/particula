"""Functions to convert mass concentrations to other concentration units."""

from numpy.typing import NDArray
import numpy as np

from particula.util.validate_inputs import validate_inputs


@validate_inputs(
    {
        "mass_concentrations": "nonnegative",
        "molar_masses": "positive",
    }
)
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


@validate_inputs(
    {
        "mass_concentrations": "nonnegative",
        "densities": "positive",
    }
)
def to_volume_fraction(
    mass_concentrations: NDArray[np.float64], densities: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Convert mass concentrations to volume fractions for N components.

    If inputs are one dimensional (1D) or floating-point scalars, the summation
    is done over the entire array. If `mass_concentrations` is 2D, the summation
    is done row-wise.

    Args:
        mass_concentrations: A list or ndarray of mass concentrations
            (SI, kg/m^3).
        densities: A list or ndarray of densities of each component
            (SI, kg/m^3).

    Returns:
        An ndarray of volume fractions. Zero volumes yield zero fractions.

    Reference:
        The volume fraction of a component is calculated by dividing the volume
        of that component (derived from mass concentration and density) by the
        total volume of all components.
        - https://en.wikipedia.org/wiki/Volume_fraction
    """
    # Calculate per-component volumes
    volumes = mass_concentrations / densities

    # Handle 1D arrays
    if volumes.ndim == 1:
        total_volume = volumes.sum()
        # If total volume is zero, return all zeros
        if total_volume == 0:
            return np.zeros_like(volumes)
        return volumes / total_volume

    # Handle 2D arrays
    elif volumes.ndim == 2:
        total_volume = volumes.sum(axis=1, keepdims=True)  # shape: (n_rows, 1)

        # Prepare an output array of the same shape
        volume_fractions = np.zeros_like(volumes)

        # We want a boolean array for which rows are nonzero
        # Squeeze to (n_rows,) for simpler indexing
        nonzero_rows = np.squeeze(total_volume != 0, axis=1)

        # Option 1: Use integer row indices
        # Identify the indices of the rows that have nonzero total volume
        indices = np.where(nonzero_rows)[0]
        # Divide row-by-row for those rows
        volume_fractions[indices, :] = (
            volumes[indices, :] / total_volume[indices, :]
        )
        return volume_fractions

    else:
        raise ValueError("mass_concentrations must be either 1D or 2D")


@validate_inputs(
    {
        "mass_concentrations": "nonnegative",
    }
)
def to_mass_fraction(
    mass_concentrations: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Convert mass concentrations to mass fractions for N components.

    If inputs are one-dimensional or float, the summation is done over the
    entire array. If `mass_concentrations` is a 2D array, the summation is
    done row-wise.

    Args:
        mass_concentrations: A list or ndarray of mass concentrations
            (SI, kg/m^3).

    Returns:
        An ndarray of mass fractions. Zero total mass yields zero fractions.

    Reference:
        The mass fraction of a component is calculated by dividing the mass
        concentration of that component by the total mass concentration of
        all components.
        - https://en.wikipedia.org/wiki/Mass_fraction_(chemistry)
    """
    # Handle 1D arrays
    if mass_concentrations.ndim == 1:
        total_mass = np.sum(mass_concentrations)
        if total_mass == 0:
            return np.zeros_like(mass_concentrations)
        return mass_concentrations / total_mass

    # Handle 2D arrays
    elif mass_concentrations.ndim == 2:
        # Row-wise sum
        total_mass = mass_concentrations.sum(axis=1, keepdims=True)
        # Prepare output
        mass_fractions = np.zeros_like(mass_concentrations)

        # Identify rows where total_mass is nonzero
        # Squeeze the mask to 1D so we can use row indices
        nonzero_rows = np.squeeze(total_mass != 0, axis=1)
        # Get actual row indices where total mass is nonzero
        row_indices = np.where(nonzero_rows)[0]

        # Compute fractions only for the nonzero rows
        mass_fractions[row_indices, :] = (
            mass_concentrations[row_indices, :] / total_mass[row_indices, :]
        )

        return mass_fractions

    else:
        raise ValueError("mass_concentrations must be either 1D or 2D")
