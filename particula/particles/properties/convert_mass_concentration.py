"""Convert mass concentrations into mole, volume, and mass fractions.

This module provides helpers for transforming component-wise mass
concentrations into normalized composition fractions for one-dimensional and
two-dimensional arrays.
"""

import numpy as np
from numpy.typing import NDArray

from particula.util.validate_inputs import validate_inputs


@validate_inputs(
    {
        "mass_concentrations": "nonnegative",
        "molar_masses": "positive",
    }
)
def get_mole_fraction_from_mass(
    mass_concentrations: NDArray[np.float64], molar_masses: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Convert mass concentrations to mole fractions.

    The mole fraction of each component is computed from the component moles,
    ``mass_concentrations / molar_masses``, normalized by the total moles in
    each composition vector. One-dimensional inputs return a single mole
    fraction vector, while two-dimensional inputs are normalized row by row.

    Args:
        mass_concentrations: Component mass concentrations in kg/m^3.
        molar_masses: Component molar masses in kg/mol.

    Returns:
        Mole fractions with the same shape as ``mass_concentrations``. If a
        total mole sum is zero, the corresponding output vector is all zeros.

    Raises:
        ValueError: If ``mass_concentrations`` is not one- or two-dimensional.
    """
    mass_concentrations = np.asarray(mass_concentrations, dtype=np.float64)
    molar_masses = np.asarray(molar_masses, dtype=np.float64)

    # Convert mass concentrations to moles for each component
    moles = mass_concentrations / molar_masses

    # Handle 1D arrays
    if isinstance(moles, float) or moles.ndim == 1:
        total_moles = np.add.reduce(moles)
        # If total moles are zero, return an array of zeros
        if total_moles == 0:
            return np.zeros_like(moles)
        return moles / total_moles

    # Handle 2D arrays
    if moles.ndim == 2:
        # Sum row-wise (shape: (n_rows, 1))
        total_moles = np.add.reduce(moles, axis=1)[:, np.newaxis]
        # Prepare output array
        mole_fractions = np.zeros_like(moles)

        # Create a row mask for nonzero total moles
        nonzero_rows = np.squeeze(total_moles != 0, axis=1)
        # Get the row indices that are nonzero
        row_indices = np.where(nonzero_rows)[0]

        # Compute fractions only for rows with nonzero total moles
        mole_fractions[row_indices, :] = (
            moles[row_indices, :] / total_moles[row_indices, :]
        )

        return mole_fractions
    raise ValueError("mass_concentrations must be either 1D or 2D")


@validate_inputs(
    {
        "mass_concentrations": "nonnegative",
        "densities": "positive",
    }
)
def get_volume_fraction_from_mass(
    mass_concentrations: NDArray[np.float64], densities: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Convert mass concentrations to volume fractions.

    The volume fraction of each component is computed from component volumes,
    ``mass_concentrations / densities``, normalized by the total volume in each
    composition vector. One-dimensional inputs return a single volume fraction
    vector, while two-dimensional inputs are normalized row by row.

    Args:
        mass_concentrations: Component mass concentrations in kg/m^3.
        densities: Component material densities in kg/m^3.

    Returns:
        Volume fractions with the same shape as ``mass_concentrations``. If a
        total volume sum is zero, the corresponding output vector is all zeros.

    Raises:
        ValueError: If ``mass_concentrations`` is not one- or two-dimensional.
    """
    mass_concentrations = np.asarray(mass_concentrations, dtype=np.float64)
    densities = np.asarray(densities, dtype=np.float64)

    # Calculate per-component volumes
    volumes = mass_concentrations / densities

    # Handle 1D arrays
    if volumes.ndim == 1:
        total_volume = np.add.reduce(volumes)
        # If total volume is zero, return all zeros
        if total_volume == 0:
            return np.zeros_like(volumes)
        return volumes / total_volume

    # Handle 2D arrays
    if volumes.ndim == 2:
        total_volume = np.add.reduce(volumes, axis=1)[:, np.newaxis]

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
    raise ValueError("mass_concentrations must be either 1D or 2D")


@validate_inputs(
    {
        "mass_concentrations": "nonnegative",
    }
)
def get_mass_fraction_from_mass(
    mass_concentrations: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Convert mass concentrations to mass fractions.

    The mass fraction of each component is computed by normalizing each
    composition vector by its total mass concentration. One-dimensional inputs
    return a single mass fraction vector, while two-dimensional inputs are
    normalized row by row.

    Args:
        mass_concentrations: Component mass concentrations in kg/m^3.

    Returns:
        Mass fractions with the same shape as ``mass_concentrations``. If a
        total mass sum is zero, the corresponding output vector is all zeros.

    Raises:
        ValueError: If ``mass_concentrations`` is not one- or two-dimensional.
    """
    # Handle 1D arrays
    if mass_concentrations.ndim == 1:
        total_mass = np.sum(mass_concentrations)
        if total_mass == 0:
            return np.zeros_like(mass_concentrations)
        return mass_concentrations / total_mass

    # Handle 2D arrays
    if mass_concentrations.ndim == 2:
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
    raise ValueError("mass_concentrations must be either 1D or 2D")
