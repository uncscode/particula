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
from particula.particles.properties.activity_module import (
    ideal_activity_molar,
    ideal_activity_volume,
    ideal_activity_mass,
    kappa_activity,
    calculate_partial_pressure,
)


class ActivityStrategy(ABC):
    """Abstract base class for vapor pressure strategies.

    This interface is used for implementing strategies based on particle
    activity calculations, specifically for calculating vapor pressures.

    Methods:
        get_name: Return the type of the activity strategy.
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

    def get_name(self) -> str:
        """Return the type of the activity strategy."""
        return self.__class__.__name__

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
        return calculate_partial_pressure(
            pure_vapor_pressure=pure_vapor_pressure,
            activity=self.activity(mass_concentration),
        )


# Ideal activity strategies
class ActivityIdealMolar(ActivityStrategy):
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
        return ideal_activity_molar(
            mass_concentration=mass_concentration, molar_mass=self.molar_mass
        )


class ActivityIdealMass(ActivityStrategy):
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
        return ideal_activity_mass(mass_concentration=mass_concentration)


class ActivityIdealVolume(ActivityStrategy):
    """Calculate ideal activity based on volume fractions.

    This strategy uses volume fractions to compute the activity, following
    the principles of Raoult's Law.

    References:
        Volume Based
            [Raoult's Law](https://en.wikipedia.org/wiki/Raoult%27s_law)
    """

    def __init__(self, density: Union[float, NDArray[np.float64]] = 0.0):
        self.density = density

    def activity(
        self, mass_concentration: Union[float, NDArray[np.float64]]
    ) -> Union[float, NDArray[np.float64]]:
        """Calculate the activity of a species based on mass concentration.

        Args:
            mass_concentration: Concentration of the species in kilograms per
                cubic meter (kg/m^3).
            density: Density of the species in kilograms per cubic meter
                (kg/m^3).

        Returns:
            Union[float, NDArray[np.float64]]: Activity of the particle,
            unitless.
        """
        return ideal_activity_volume(
            mass_concentration=mass_concentration, density=self.density
        )


# Non-ideal activity strategies
class ActivityKappaParameter(ActivityStrategy):
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
        self.kappa = kappa
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
        return kappa_activity(
            mass_concentration=mass_concentration,
            kappa=self.kappa,
            density=self.density,
            molar_mass=self.molar_mass,
            water_index=self.water_index,
        )
