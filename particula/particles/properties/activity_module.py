"""Activity Functions Module."""


from typing import Union
import numpy as np
from numpy.typing import NDArray
from particula.util.converting import convert_mass_concentration


def ideal_activity_molar(
    mass_concentration: Union[float, NDArray[np.float64]],
    molar_mass: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate the ideal activity of a species based on mole fractions.

    This function computes the activity based on the mole fractions of species
    according to Raoult's Law.

    Args:
        mass_concentration (float or NDArray[np.float64]): Mass concentration
            of the species in kilograms per cubic meter (kg/m^3).
        molar_mass (float or NDArray[np.float64]): Molar mass of the species in
            kilograms per mole (kg/mol). A single value applies to all species
            if only one is provided.

    Returns:
        float or NDArray[np.float64]: Activity of the species, unitless.

    Example:
        ``` py title="Example"
        ideal_activity_molar(
            mass_concentration=np.array([1.0, 2.0]),
            molar_mass=np.array([18.015, 28.97]))
        # array([0.0525, 0.0691])
        ```

    References:
        Raoult's Law: https://en.wikipedia.org/wiki/Raoult%27s_law
    """
    if isinstance(mass_concentration, float):
        return 1.0
    return convert_mass_concentration.to_mole_fraction(
        mass_concentrations=mass_concentration,  # type: ignore
        molar_masses=molar_mass  # type: ignore
    )


def ideal_activity_volume(
    mass_concentration: Union[float, NDArray[np.float64]],
    density: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate the ideal activity of a species based on volume fractions.

    This function computes the activity based on the volume fractions of
    species consistent with Raoult's Law.

    Args:
        mass_concentration (float or NDArray[np.float64]): Mass concentration
            of the species in kilograms per cubic meter (kg/m^3).
        density (float or NDArray[np.float64]): Density of the species in
            kilograms per cubic meter (kg/m^3). A single value applies to all
            species if only one is provided.

    Returns:
        float or NDArray[np.float64]: Activity of the species, unitless.

    Example:
        ``` py title="Example"
        ideal_activity_volume(
            mass_concentration=np.array([1.0, 2.0]),
            density=np.array([1000.0, 1200.0]))
        # array([0.001, 0.002])
        ```

    References:
        Raoult's Law: https://en.wikipedia.org/wiki/Raoult%27s_law
    """
    if isinstance(mass_concentration, float):
        return 1.0
    return convert_mass_concentration.to_volume_fraction(
        mass_concentrations=mass_concentration,  # type: ignore
        densities=density  # type: ignore
    )


def ideal_activity_mass(
    mass_concentration: Union[float, NDArray[np.float64]]
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate the ideal activity of a species based on mass fractions.

    This function computes the activity based on the mass fractions of species
    consistent with Raoult's Law.

    Args:
        mass_concentration (float or NDArray[np.float64]): Mass concentration
            of the species in kilograms per cubic meter (kg/m^3).

    Returns:
        float or NDArray[np.float64]]: Activity of the species, unitless.

    Example:
        ``` py title="Example"
        ideal_activity_mass(np.array([1.0, 2.0]))
        array([0.3333, 0.6667])
        ```

    References:
        Raoult's Law: https://en.wikipedia.org/wiki/Raoult%27s_law
    """
    if isinstance(mass_concentration, float):
        return 1.0
    return convert_mass_concentration.to_mass_fraction(
        mass_concentrations=mass_concentration  # type: ignore
    )


# pylint: disable=too-many-locals
def kappa_activity(
    mass_concentration: NDArray[np.float64],
    kappa: NDArray[np.float64],
    density: NDArray[np.float64],
    molar_mass: NDArray[np.float64],
    water_index: int,
) -> NDArray[np.float64]:
    """
    Calculate the activity of species based on the kappa hygroscopic parameter.

    This function computes the activity using the kappa parameter and the
    species' mass concentrations, considering the volume fractions and water
    content.

    Args:
        mass_concentration (float or NDArray[np.float64]]): Mass concentration
            of the species in kilograms per cubic meter (kg/m^3).
        kappa (NDArray[np.float64]): Kappa hygroscopic parameter, unitless.
        density (NDArray[np.float64]): Density of the species in kilograms per
            cubic meter (kg/m^3).
        molar_mass (NDArray[np.float64]): Molar mass of the species in
            kilograms per mole (kg/mol).
        water_index (int): Index of water in the mass concentration array.

    Returns:
        NDArray[np.float64]: Activity of the species, unitless.

    Example:
        ``` py title="Example"
        kappa_activity(
            mass_concentration=np.array([[1.0, 2.0], [3.0, 4.0]]),
            kappa=np.array([0.0, 0.2]),
            density=np.array([1000.0, 1200.0]),
            molar_mass=np.array([18.015, 28.97]),
            water_index=0
        )
        # array([[0.95, 0.75], [0.85, 0.65]])
        ```

    References:
        Petters, M. D., & Kreidenweis, S. M. (2007). A single parameter
        representation of hygroscopic growth and cloud condensation nucleus
        activity. Atmospheric Chemistry and Physics, 7(8), 1961-1971.
        DOI: https://doi.org/10.5194/acp-7-1961-2007
    """

    volume_fractions = convert_mass_concentration.to_volume_fraction(
        mass_concentrations=mass_concentration,
        densities=density)
    # other species activity based on mole fraction
    activity = convert_mass_concentration.to_mole_fraction(
        mass_concentrations=mass_concentration,
        molar_masses=molar_mass
    )

    expanded = False
    if volume_fractions.ndim == 1:
        volume_fractions = volume_fractions[np.newaxis, :]
        expanded = True

    # water activity based on kappa
    water_volume_fraction = volume_fractions[:, water_index]
    solute_volume_fractions = np.delete(
        volume_fractions, water_index, axis=1
    )
    kappa = np.delete(kappa, water_index)
    # volume weighted kappa, EQ 7 Petters and Kreidenweis (2007)
    if solute_volume_fractions.shape[1] == 1 and not expanded:
        kappa_weighted = np.full_like(water_volume_fraction, kappa)
    else:
        solute_volume_fractions = np.divide(
            solute_volume_fractions,
            np.sum(solute_volume_fractions, axis=1, keepdims=True),
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
        activity[water_index] = water_activity
        return activity

    activity[:, water_index] = water_activity
    return activity


def calculate_partial_pressure(
    pure_vapor_pressure: Union[float, NDArray[np.float64]],
    activity: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate the partial pressure of a species based on its activity and pure
        vapor pressure.

    Args:
        pure_vapor_pressure (float or NDArray[np.float64]): Pure vapor pressure
            of the species in pascals (Pa).
        activity (float or NDArray[np.float64]): Activity of the species,
            unitless.

    Returns:
        float or NDArray[np.float64]: Partial pressure of the species in
        pascals (Pa).

    Example:
        ``` py title="Example"
        calculate_partial_pressure(1000.0, 0.95)
        # 950.0
        ```
    """
    return pure_vapor_pressure * activity
