"""Functions to convert mass concentrations to other concentration units."""

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
    """Convert mass concentrations to mole fractions for N components.

     The mole fraction is computed using:

     - xᵢ = (mᵢ / Mᵢ) / Σⱼ(mⱼ / Mⱼ)
         - xᵢ is the mole fraction of component i,
         - mᵢ is the mass concentration of component i (kg/m³),
         - Mᵢ is the molar mass of component i (kg/mol).

    Arguments:
         - mass_concentrations : Mass concentrations (kg/m³). Can be 1D or 2D.
         - molar_masses : Molar masses (kg/mol). Must match dimensions of
           mass_concentrations.

    Returns:
         - Mole fractions (unitless). Rows sum to 1 if input is 2D; returns 1D
           mole fractions if input is 1D.

    Examples:
         ```py
         import numpy as np
         import particula as par
         mass_conc = np.array([0.2, 0.8])  # kg/m³
         mol_masses = np.array([0.018, 0.032])  # kg/mol
         get_mole_fraction_from_mass(mass_conc, mol_masses))
         # Output might be array([0.379..., 0.620...])
         ```

    References:
         - Wikipedia contributors, "Mole fraction," Wikipedia,
           https://en.wikipedia.org/wiki/Mole_fraction.
    """
    # Convert mass concentrations to moles for each component
    moles = mass_concentrations / molar_masses

    # Handle 1D arrays
    if isinstance(moles, float) or moles.ndim == 1:
        total_moles = np.sum(moles)
        # If total moles are zero, return an array of zeros
        if total_moles == 0:
            return np.zeros_like(moles)
        return moles / total_moles

    # Handle 2D arrays
    if moles.ndim == 2:
        # Sum row-wise (shape: (n_rows, 1))
        total_moles = moles.sum(axis=1, keepdims=True)
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
    """Convert mass concentrations to volume fractions for N components.

    The volume fraction is determined by:

    - ϕᵢ = vᵢ / vₜₒₜₐₗ
        - ϕᵢ is the volume fraction of component i (unitless),
        - vᵢ is the volume of component i (m³),
        - vₜₒₜₐₗ is the total volume of all components (m³).

    Volumes computed from mass concentration (mᵢ) and density (ρᵢ) using:
    - vᵢ = mᵢ / ρᵢ.

    Arguments:
        - mass_concentrations : Mass concentrations (kg/m³). Can be 1D or 2D.
        - densities : Densities (kg/m³). Must match the shape of
          mass_concentrations.

    Returns:
        - Volume fractions (unitless). Rows sum to 1 if input is 2D; returns 1D
          volume fractions if input is 1D.

    Examples:
        ```py
        import numpy as np
        import particula as par

        mass_conc = np.array([[1.0, 2.0], [0.5, 0.5]])  # kg/m³
        dens = np.array([1000.0, 800.0])               # kg/m³
        par.get_volume_fraction_from_mass(mass_conc, dens))
        # Output:
        # array([[0.444..., 0.555...],
        #        [0.5     , 0.5     ]])
        ```

    References:
        - Wikipedia contributors, "Volume fraction," Wikipedia,
          https://en.wikipedia.org/wiki/Volume_fraction.
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
    if volumes.ndim == 2:
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
    raise ValueError("mass_concentrations must be either 1D or 2D")


@validate_inputs(
    {
        "mass_concentrations": "nonnegative",
    }
)
def get_mass_fraction_from_mass(
    mass_concentrations: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Convert mass concentrations to mass fractions for N components.

    The mass fraction is computed by:

    - wᵢ = mᵢ / mₜₒₜₐₗ
        - wᵢ is the mass fraction of component i (unitless),
        - mᵢ is the mass concentration of component i (kg/m³),
        - mₜₒₜₐₗ is the total mass concentration of all components (kg/m³).

    Arguments:
        - mass_concentrations : Mass concentrations (kg/m³). Can be 1D or 2D.

    Returns:
        - Mass fractions (unitless). Rows sum to 1 if input is 2D; returns 1D
          mass fractions if input is 1D.

    Examples:
        ```py
        import numpy as np
        import particula as par

        mass_conc = np.array([10.0, 30.0, 60.0])  # kg/m³
        par.get_mass_fraction(mass_conc)
        # Output might be array([0.1, 0.3, 0.6])
        ```

    References:
        - Wikipedia contributors, "Mass fraction (chemistry)," Wikipedia,
          https://en.wikipedia.org/wiki/Mass_fraction_(chemistry).
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
