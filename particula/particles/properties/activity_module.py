"""Activity Functions Module."""

from typing import Union

import numpy as np
from numpy.typing import NDArray

from particula.particles.properties.convert_mass_concentration import (
    get_mass_fraction_from_mass,
    get_mole_fraction_from_mass,
    get_volume_fraction_from_mass,
)
from particula.util.validate_inputs import validate_inputs


@validate_inputs(
    {
        "mass_concentration": "nonnegative",
        "molar_mass": "positive",
    }
)
def get_ideal_activity_molar(
    mass_concentration: Union[float, NDArray[np.float64]],
    molar_mass: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """Compute the ideal activity based on mole fractions.

    This function calculates the activity of a species using its mole fraction,
    which follows Raoult's Law. The ideal activity (aᵢ) is determined using:

    - aᵢ = Xᵢ
        - Xᵢ is the mole fraction of species i.

    Arguments:
        - mass_concentration : Mass concentration of the species in kg/m³.
        - molar_mass : Molar mass of the species in kg/mol.

    Returns:
        - Ideal activity of the species as a dimensionless value.

    Examples:
        ``` py title="Example"
        import particula as par
        par.particles.get_ideal_activity_molar(
            mass_concentration=np.array([1.0, 2.0]),
            molar_mass=np.array([18.015, 28.97])
        )
        # Output: array([...])
        ```

    References:
        - Raoult's Law, "Raoult's law," Wikipedia,
          https://en.wikipedia.org/wiki/Raoult%27s_law.
    """
    if isinstance(mass_concentration, float):
        return 1.0
    return get_mole_fraction_from_mass(
        mass_concentrations=mass_concentration,  # type: ignore
        molar_masses=molar_mass,  # type: ignore
    )


@validate_inputs(
    {
        "mass_concentration": "nonnegative",
        "density": "positive",
    }
)
def get_ideal_activity_volume(
    mass_concentration: Union[float, NDArray[np.float64]],
    density: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """Compute the ideal activity based on volume fractions.

    This function calculates the activity of a species using its volume
    fraction. In an ideal mixture, the activity (aᵢ) can be expressed as:

    - aᵢ = φᵢ
        - φᵢ is the volume fraction of species i.

    Arguments:
        - mass_concentration : Mass concentration of the species in kg/m³.
        - density : Density of the species in kg/m³.

    Returns:
        - Ideal activity of the species as a dimensionless value.

    Examples:
        ``` py title="Example"
        import particula as par
        par.particles.get_ideal_activity_volume(
            mass_concentration=np.array([1.0, 2.0]),
            density=np.array([1000.0, 1200.0])
        )
        # Output: array([...])
        ```

    References:
        - Raoult's Law, "Raoult's law," Wikipedia,
          https://en.wikipedia.org/wiki/Raoult%27s_law.
    """
    if isinstance(mass_concentration, float):
        return 1.0
    return get_volume_fraction_from_mass(
        mass_concentrations=mass_concentration,  # type: ignore
        densities=density,  # type: ignore
    )


@validate_inputs(
    {
        "mass_concentration": "nonnegative",
    }
)
def get_ideal_activity_mass(
    mass_concentration: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """Compute the ideal activity based on mass fractions.

    This function calculates the activity of a species using its mass fraction.
    In an ideal mixture, the activity (aᵢ) can be expressed as:

    - aᵢ = wᵢ
        - wᵢ is the mass fraction of species i.

    Arguments:
        - mass_concentration : Mass concentration of the species in kg/m³.

    Returns:
        - Ideal activity of the species as a dimensionless value.

    Examples:
        ``` py title="Example"
        import particula as par
        par.particles.get_ideal_activity_mass(np.array([1.0, 2.0]))
        # Output: array([...])
        ```

    References:
        - Raoult's Law, "Raoult's law," Wikipedia,
          https://en.wikipedia.org/wiki/Raoult%27s_law.
    """
    if isinstance(mass_concentration, float):
        return 1.0
    return get_mass_fraction_from_mass(
        mass_concentrations=mass_concentration  # type: ignore
    )


@validate_inputs(
    {
        "mass_concentration": "nonnegative",
        "kappa": "nonnegative",
        "density": "positive",
        "molar_mass": "positive",
    }
)
def get_kappa_activity(
    mass_concentration: NDArray[np.float64],
    kappa: NDArray[np.float64],
    density: NDArray[np.float64],
    molar_mass: NDArray[np.float64],
    water_index: int,
) -> NDArray[np.float64]:
    # pylint: disable=too-many-locals
    """Compute species activity using κ hygroscopic growth parameter.

    This function calculates the activity of a mixture by combining
    volume-fraction weighted κ-values. The water activity (aₘ) is
    determined by:

    - aₘ = 1 / (1 + κₑ ( Vₛ / Vₐ ))
        - κₑ is the volume-fraction weighted hygroscopic parameter.
        - Vₛ is the total solute volume fraction (all species except water).
        - Vₐ is the water volume fraction.

    Arguments:
        - mass_concentration : Array of mass concentrations in kg/m³.
        - kappa : Array of κ (kappa) hygroscopic parameters, dimensionless.
        - density : Array of densities in kg/m³ for each species.
        - molar_mass : Array of molar masses in kg/mol for each species.
        - water_index : Index of the water component in the arrays.

    Returns:
        - Array of species activities, dimensionless.

    Examples:
        ``` py title="Example"
        import particula as par
        par.particles.get_kappa_activity(
            mass_concentration=np.array([[1.0, 2.0], [3.0, 4.0]]),
            kappa=np.array([0.0, 0.2]),
            density=np.array([1000.0, 1200.0]),
            molar_mass=np.array([18.015, 28.97]),
            water_index=0
        )
        # Output: array([...])
        ```

    References:
        - Petters, M. D., & Kreidenweis, S. M. (2007). "A single parameter
          representation of hygroscopic growth and cloud condensation nucleus
          activity," Atmospheric Chemistry and Physics, 7(8), 1961-1971.
          DOI: https://doi.org/10.5194/acp-7-1961-2007.
    """
    volume_fractions = get_volume_fraction_from_mass(
        mass_concentrations=mass_concentration, densities=density
    )
    # other species activity based on mole fraction
    activity = get_mole_fraction_from_mass(
        mass_concentrations=mass_concentration, molar_masses=molar_mass
    )

    expanded = False
    if volume_fractions.ndim == 1:
        volume_fractions = volume_fractions[np.newaxis, :]
        expanded = True

    # water activity based on kappa
    water_volume_fraction = volume_fractions[:, water_index]
    solute_volume_fractions = np.delete(volume_fractions, water_index, axis=1)
    kappa = np.delete(kappa, water_index)
    # volume weighted kappa, EQ 7 Petters and Kreidenweis (2007)
    if solute_volume_fractions.shape[1] == 1 and not expanded:
        kappa_weighted = np.full_like(water_volume_fraction, kappa)
    else:
        solute_volume_fractions = np.divide(
            solute_volume_fractions,
            np.sum(solute_volume_fractions, axis=1, keepdims=True),
            out=np.zeros_like(solute_volume_fractions),
            where=np.sum(solute_volume_fractions, axis=1, keepdims=True) != 0,
        )
        kappa_weighted = np.sum(solute_volume_fractions * kappa, axis=1)
    # kappa activity parameterization, EQ 2 Petters and Kreidenweis
    # (2007)
    solute_volume = 1 - water_volume_fraction
    numerator = kappa_weighted * solute_volume
    denominator = water_volume_fraction
    volume_term = np.divide(
        numerator,
        denominator,
        out=-np.ones_like(denominator),  # Set when condition is false
        where=denominator > 0,
    )
    water_activity = np.divide(
        1,
        1 + volume_term,
        out=np.zeros_like(volume_term),  # For zero water volume fraction
        where=volume_term != -1,
    )

    # Replace water activity with kappa activity
    if expanded:
        activity[water_index] = water_activity[0]
        return activity

    activity[:, water_index] = water_activity
    return activity


@validate_inputs(
    {
        "pure_vapor_pressure": "positive",
        "activity": "nonnegative",
    }
)
def get_surface_partial_pressure(
    pure_vapor_pressure: Union[float, NDArray[np.float64]],
    activity: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """Compute the partial pressure from activity and pure vapor pressure.

    This function calculates the partial pressure (pᵢ) of a species, given its
    activity (aᵢ) and pure vapor pressure (pᵢ*). It follows:

    - pᵢ = aᵢ × pᵢ*

    Arguments:
        - pure_vapor_pressure : Pure vapor pressure of the species in Pa.
        - activity : Activity of the species, dimensionless.

    Returns:
        - Partial pressure of the species in Pa.

    Examples:
        ``` py title="Example"
        import particula as par
        par.particles.get_surface_partial_pressure(1000.0, 0.95)
        # Output: 950.0
        ```
    """
    return pure_vapor_pressure * activity
