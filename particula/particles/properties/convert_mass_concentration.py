"""Convert mass concentrations into mole, volume, and mass fractions.

This module provides helpers for transforming component-wise mass
concentrations into normalized composition fractions for one-dimensional and
two-dimensional arrays.
"""

import numpy as np
from numpy.typing import NDArray

from particula.util.validate_inputs import validate_inputs


def _normalize_rows(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Normalize 2D rows while leaving zero-total rows as zeros."""
    row_totals = np.add.reduce(values, axis=1, keepdims=True)
    normalized = np.zeros_like(values)
    np.divide(values, row_totals, out=normalized, where=row_totals != 0)
    return normalized


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

    if moles.ndim == 0:
        if moles == 0:
            return np.zeros_like(moles)
        return moles / moles

    # Handle 1D arrays
    if moles.ndim == 1:
        total_moles = np.add.reduce(moles)
        # If total moles are zero, return an array of zeros
        if total_moles == 0:
            return np.zeros_like(moles)
        return moles / total_moles

    # Handle 2D arrays
    if moles.ndim == 2:
        return _normalize_rows(moles)
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

    if volumes.ndim == 0:
        if volumes == 0:
            return np.zeros_like(volumes)
        return volumes / volumes

    # Handle 1D arrays
    if volumes.ndim == 1:
        total_volume = np.add.reduce(volumes)
        # If total volume is zero, return all zeros
        if total_volume == 0:
            return np.zeros_like(volumes)
        return volumes / total_volume

    # Handle 2D arrays
    if volumes.ndim == 2:
        return _normalize_rows(volumes)
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
    mass_concentrations = np.asarray(mass_concentrations, dtype=np.float64)

    if mass_concentrations.ndim == 0:
        if mass_concentrations == 0:
            return np.zeros_like(mass_concentrations)
        return mass_concentrations / mass_concentrations

    # Handle 1D arrays
    if mass_concentrations.ndim == 1:
        total_mass = np.sum(mass_concentrations)
        if total_mass == 0:
            return np.zeros_like(mass_concentrations)
        return mass_concentrations / total_mass

    # Handle 2D arrays
    if mass_concentrations.ndim == 2:
        return _normalize_rows(mass_concentrations)
    raise ValueError("mass_concentrations must be either 1D or 2D")
