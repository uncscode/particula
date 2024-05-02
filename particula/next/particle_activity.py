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

# particula imports
from particula.util.convert import (
    mass_concentration_to_mole_fraction,
    mass_concentration_to_volume_fraction
    )


# Abstract class for vapor pressure strategies
class ParticleActivityStrategy(ABC):
    """
    Abstract class for implementing vapor pressure strategies based on
    particle activity calculations.
    """

    @abstractmethod
    def activity(
                self, mass_concentration: Union[float, NDArray[np.float_]]
            ) -> Union[float, NDArray[np.float_]]:
        """
        Calculate the activity of a species based on its mass concentration.

        Args:
        - mass_concentration (float or NDArray[float]): Concentration of the
        species [kg/m^3]

        Returns:
        - float or NDArray[float]: Activity of the particle, unitless.
        """

    def partial_pressure(
        self,
        pure_vapor_pressure: Union[float, NDArray[np.float_]],
        mass_concentration: Union[float, NDArray[np.float_]]
    ) -> Union[float, NDArray[np.float_]]:
        """
        Calculate the vapor pressure of species in the particle phase based on
        activity.

        Args:
        - pure_vapor_pressure (float or NDArray[float]): Pure vapor pressure
        of the species [Pa]
        - mass_concentration (float or NDArray[float]): Concentration of the
        species [kg/m^3]

        Returns:
        - float or NDArray[float]: Vapor pressure of the particle [Pa].
        """
        return pure_vapor_pressure * self.activity(mass_concentration)


# Ideal activity strategies
class MolarIdealActivity(ParticleActivityStrategy):
    """Ideal activity strategy, based on mole fractions.

    Keyword arguments:
    ------------------
    - molar_mass (Union[float, NDArray[np.float_]]): Molar mass of the species
    [kg/mol]. If a single value is provided, it will be used for all species.

    References:
    -----------
    - Molar Based Raoult's Law https://en.wikipedia.org/wiki/Raoult%27s_law
    """

    def __init__(
        self,
        molar_mass: Union[float, NDArray[np.float_]] = 0.0
    ):
        self.molar_mass = molar_mass

    def activity(
                self,
                mass_concentration: Union[float, NDArray[np.float_]]
            ) -> Union[float, NDArray[np.float_]]:
        """Calculate the activity of a species.

        Args:
        -----
        - mass_concentration (float): Concentration of the species [kg/m^3]

        Returns:
        --------
        - float: Activity of the particle [unitless].
        """
        # return for single species, activity is always 1
        if isinstance(mass_concentration, float):
            return 1.0
        # multiple species, calculate mole fractions
        mole_fraction = mass_concentration_to_mole_fraction(
            mass_concentrations=mass_concentration,
            molar_masses=self.molar_mass
        )
        return mole_fraction


class MassIdealActivity(ParticleActivityStrategy):
    """Ideal activity strategy, based on mass fractions.

    Keyword arguments:
    ------------------
    - None needed

    References:
    -----------
    - Mass Based Raoult's Law https://en.wikipedia.org/wiki/Raoult%27s_law
    """

    def activity(
                self,
                mass_concentration: Union[float, NDArray[np.float_]]
            ) -> Union[float, NDArray[np.float_]]:
        """Calculate the activity of a species.

        Args:
        -----
        - mass_concentration (float): Concentration of the species [kg/m^3]

        Returns:
        --------
        - float: Activity of the particle [unitless].
        """
        # return for single species, activity is always 1
        if isinstance(mass_concentration, float):
            return 1.0
        # multiple species, calculate mass fractions
        mass_fraction = mass_concentration / np.sum(mass_concentration)
        return mass_fraction


# Non-ideal activity strategies
class KappaParameterActivity(ParticleActivityStrategy):
    """Non-ideal activity strategy, based on kappa hygroscopic parameter for
    non-ideal water, and mole fraction for other species.

    Keyword arguments:
    ------------------
    - kappa (NDArray[np.float_]): Kappa hygroscopic parameter [unitless],
    include a value for water (that will be removed in the calculation).
    - density (NDArray[np.float_]): Density of the species [kg/m^3].
    - molar_mass (NDArray[np.float_]): Molar mass of the species [kg/mol].
    - water_index (int): Index of water in the mass_concentration array.
    """

    def __init__(
        self,
        kappa: NDArray[np.float_] = np.array([0.0], dtype=np.float_),
        density: NDArray[np.float_] = np.array([0.0], dtype=np.float_),
        molar_mass: NDArray[np.float_] = np.array([0.0], dtype=np.float_),
        water_index: int = 0,
    ):
        self.kappa = np.delete(kappa, water_index)  # maybe change this later
        self.density = density
        self.molar_mass = molar_mass
        self.water_index = water_index

    def activity(
                self,
                mass_concentration: Union[float, NDArray[np.float_]]
            ) -> Union[float, NDArray[np.float_]]:
        """Calculate the activity of a species.

        Args:
        -----
        - mass_concentration (float): Concentration of the species [kg/m^3]

        Returns:
        --------
        - float: Activity of the particle [unitless].

        References:
        -----------
        Petters, M. D., & Kreidenweis, S. M. (2007). A single parameter
        representation of hygroscopic growth and cloud condensation nucleus
        activity. Atmospheric Chemistry and Physics, 7(8), 1961-1971.
        https://doi.org/10.5194/acp-7-1961-2007
        EQ 2 and 7
        """
        volume_fractions = mass_concentration_to_volume_fraction(
            mass_concentrations=mass_concentration,
            densities=self.density
        )
        water_volume_fraction = volume_fractions[self.water_index]
        solute_volume_fractions = np.delete(volume_fractions, self.water_index)
        solute_volume = 1-water_volume_fraction
        # volume weighted kappa, EQ 7 Petters and Kreidenweis (2007)
        kappa_weighted = np.sum(
            solute_volume_fractions/solute_volume
            * self.kappa
        )
        # kappa activity parameterization, EQ 2 Petters and Kreidenweis (2007)
        water_activity = (
            1 + kappa_weighted * solute_volume/water_volume_fraction)**(-1)
        # other species activity based on mole fraction
        activity = mass_concentration_to_mole_fraction(
            mass_concentrations=mass_concentration,
            molar_masses=self.molar_mass
        )
        # replace water activity with kappa activity
        activity[self.water_index] = water_activity
        return activity



# Factory function for creating activity strategies
def particle_activity_strategy_factory(
            strategy_type: str,
            **kwargs: dict  # type: ignore
        ):
    """
    Factory function for creating activity strategies. Used for calculating
    activity and partial pressure of species in a mixture of liquids.

    Args:
    - strategy_type (str): Type of activity strategy to use. The options are:
        - molar_ideal: Ideal activity based on mole fractions.
        - mass_ideal: Ideal activity based on mass fractions.
        - kappa: Non-ideal activity based on kappa hygroscopic parameter.
    - kwargs: Arguments for the activity strategy."""
    if strategy_type == "molar_ideal":
        return MolarIdealActivity(**kwargs)
    if strategy_type == "mass_ideal":
        return MassIdealActivity()
    if strategy_type == "kappa":
        return KappaParameterActivity(**kwargs)
    raise ValueError("Unknown strategy type")
