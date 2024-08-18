"""
Class strategies for activities and vapor pressure over mixture of liquids
surface Using Raoult's Law, and strategies ideal, non-ideal, kappa hygroscopic
parameterizations.
"""

# pyright: reportArgumentType=false

from abc import ABC, abstractmethod
from typing import Union
from numpy.typing import NDArray
import numpy as np
from particula.util.convert import (
    mass_concentration_to_mole_fraction,
    mass_concentration_to_volume_fraction,
)


class ActivityStrategy(ABC):
    """Abstract base class for vapor pressure strategies.

    This interface is used for implementing strategies based on particle
    activity calculations, specifically for calculating vapor pressures.

    Methods:
        activity: Calculate the activity of a species.
        partial_pressure: Calculate the partial pressure of a species in
            the mixture.
    """

    @abstractmethod
    def activity(
        self, mass_concentration: Union[float, NDArray[np.float64]]
    ) -> Union[float, NDArray[np.float64]]:
        """Calculate the activity of a species based on its mass concentration.

        Args:
            mass_concentration: Concentration of the species [kg/m^3]

        Returns:
            float or NDArray[float]: Activity of the particle, unitless.
        """

    def partial_pressure(
        self,
        pure_vapor_pressure: Union[float, NDArray[np.float64]],
        mass_concentration: Union[float, NDArray[np.float64]],
    ) -> Union[float, NDArray[np.float64]]:
        """Calculate the vapor pressure of species in the particle phase.

        This method computes the vapor pressure based on the species' activity
        considering its pure vapor pressure and mass concentration.

        Args:
            pure_vapor_pressure: Pure vapor pressure of the species in
            pascals (Pa).
            mass_concentration: Concentration of the species in kilograms per
            cubic meter (kg/m^3).

        Returns:
            Union[float, NDArray[np.float64]]: Vapor pressure of the particle
            in pascals (Pa).
        """
        return pure_vapor_pressure * self.activity(mass_concentration)


# Ideal activity strategies
class IdealActivityMolar(ActivityStrategy):
    """Calculate ideal activity based on mole fractions.

    This strategy uses mole fractions to compute the activity, adhering to
    the principles of Raoult's Law.

    Args:
        molar_mass (Union[float, NDArray[np.float64]]): Molar mass of the
        species [kg/mol]. A single value applies to all species if only one
        is provided.

    References:
        Molar [Raoult's Law](https://en.wikipedia.org/wiki/Raoult%27s_law)
    """

    def __init__(self, molar_mass: Union[float, NDArray[np.float64]] = 0.0):
        self.molar_mass = molar_mass

    def activity(
        self, mass_concentration: Union[float, NDArray[np.float64]]
    ) -> Union[float, NDArray[np.float64]]:
        """Calculate the activity of a species based on mass concentration.

        Args:
            mass_concentration: Concentration of the species in kilograms per
            cubic meter (kg/m^3).

        Returns:
            Union[float, NDArray[np.float64]]: Activity of the species,
            unitless.
        """

        # return for single species, activity is always 1
        if isinstance(mass_concentration, float):
            return 1.0
        # multiple species, calculate mole fractions
        return mass_concentration_to_mole_fraction(
            mass_concentrations=mass_concentration,
            molar_masses=self.molar_mass,
        )


class IdealActivityMass(ActivityStrategy):
    """Calculate ideal activity based on mass fractions.

    This strategy utilizes mass fractions to determine the activity, consistent
    with the principles outlined in Raoult's Law.

    References:
        Mass Based [Raoult's Law](https://en.wikipedia.org/wiki/Raoult%27s_law)
    """

    def activity(
        self, mass_concentration: Union[float, NDArray[np.float64]]
    ) -> Union[float, NDArray[np.float64]]:
        """Calculate the activity of a species based on mass concentration.

        Args:
            mass_concentration: Concentration of the species in kilograms
            per cubic meter (kg/m^3).

        Returns:
            Union[float, NDArray[np.float64]]: Activity of the particle,
            unitless.
        """

        # return for single species, activity is always 1
        if isinstance(mass_concentration, float):
            return 1.0
        return mass_concentration / np.sum(mass_concentration)


# Non-ideal activity strategies
class KappaParameterActivity(ActivityStrategy):
    """Non-ideal activity strategy based on the kappa hygroscopic parameter.

    This strategy calculates the activity using the kappa hygroscopic
    parameter, a measure of hygroscopicity. The activity is determined by the
    species' mass concentration along with the hygroscopic parameter.

    Args:
        kappa: Kappa hygroscopic parameter, unitless.
            Includes a value for water which is excluded in calculations.
        density: Density of the species in kilograms per
            cubic meter (kg/m^3).
        molar_mass: Molar mass of the species in kilograms
            per mole (kg/mol).
        water_index: Index of water in the mass concentration array.
    """

    def __init__(
        self,
        kappa: NDArray[np.float64] = np.array([0.0], dtype=np.float64),
        density: NDArray[np.float64] = np.array([0.0], dtype=np.float64),
        molar_mass: NDArray[np.float64] = np.array([0.0], dtype=np.float64),
        water_index: int = 0,
    ):
        self.kappa = np.delete(kappa, water_index)  # maybe change this later
        self.density = density
        self.molar_mass = molar_mass
        self.water_index = water_index

    def activity(
        self, mass_concentration: Union[float, NDArray[np.float64]]
    ) -> Union[float, NDArray[np.float64]]:
        """Calculate the activity of a species based on mass concentration.

        Args:
            mass_concentration: Concentration of the species in kilograms per
            cubic meter (kg/m^3).

        Returns:
            Union[float, NDArray[np.float64]]: Activity of the particle,
            unitless.

        References:
            Petters, M. D., & Kreidenweis, S. M. (2007). A single parameter
            representation of hygroscopic growth and cloud condensation nucleus
            activity. Atmospheric Chemistry and Physics, 7(8), 1961-1971.
            [DOI](https://doi.org/10.5194/acp-7-1961-2007), see EQ 2 and 7.
        """

        volume_fractions = mass_concentration_to_volume_fraction(
            mass_concentrations=mass_concentration, densities=self.density
        )
        if isinstance(mass_concentration, np.ndarray) and (
            mass_concentration.ndim == 2
        ):
            water_volume_fraction = volume_fractions[:, self.water_index]
            solute_volume_fractions = np.delete(
                volume_fractions, self.water_index, axis=1
            )
            solute_volume = 1 - water_volume_fraction
            # volume weighted kappa, EQ 7 Petters and Kreidenweis (2007)
            kappa_weighted = np.sum(
                solute_volume_fractions / solute_volume * self.kappa, axis=1
            )
            # kappa activity parameterization, EQ 2 Petters and Kreidenweis (2007)
            water_activity = (
                1 + kappa_weighted * solute_volume / water_volume_fraction
            ) ** (-1)
            # other species activity based on mole fraction
            activity = mass_concentration_to_mole_fraction(
                mass_concentrations=mass_concentration,
                molar_masses=self.molar_mass,
            )
            # replace water activity with kappa activity
            activity[:, self.water_index] = water_activity
            return activity
        # single species
        water_volume_fraction = volume_fractions[self.water_index]
        solute_volume_fractions = np.delete(volume_fractions, self.water_index)
        solute_volume = 1 - water_volume_fraction
        # volume weighted kappa, EQ 7 Petters and Kreidenweis (2007)
        kappa_weighted = np.sum(
            solute_volume_fractions / solute_volume * self.kappa
        )
        # kappa activity parameterization, EQ 2 Petters and Kreidenweis (2007)
        water_activity = (
            1 + kappa_weighted * solute_volume / water_volume_fraction
        ) ** (-1)
        # other species activity based on mole fraction
        activity = mass_concentration_to_mole_fraction(
            mass_concentrations=mass_concentration,
            molar_masses=self.molar_mass,
        )
        # replace water activity with kappa activity
        activity[self.water_index] = water_activity
        return activity
