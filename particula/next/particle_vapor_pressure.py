"""
Class strategies for caculating vapor pressure over mixture of liquids surface
Using Raoult's Law, and strategies ideal, non-ideal, kappa hygroscopic
parameterizations."""

from abc import ABC, abstractmethod
from typing import Union
from numpy.typing import NDArray
import numpy as np

# particula imports
from particula.constants import GAS_CONSTANT


# Abstract class for vapor pressure strategies
class ParticleVaporPressure(ABC):
    """Abstract class for vapor pressure strategies."""

    def calculate_partial_pressure(
                self,
                pure_vapor_pressures: Union[float, NDArray[np.float_]],
                activities: Union[float, NDArray[np.float_]]
            ) -> Union[float, NDArray[np.float_]]:
        """Calculate the vapor pressure of species in the particle phase.

        Args:
        -----
        - pure_vapor_pressure (float): Pure vapor pressure of the species [Pa]
        - activity (float): Activity of the species. [unitless]

        Returns:
        - float: Vapor pressure of the particle [Pa].
        """
        return pure_vapor_pressure * activity

    @abstractmethod
    def calculate_activity(
                self,
                mass_concentrations: Union[float, NDArray[np.float_]]
            ) -> Union[float, NDArray[np.float_]]:
        """Calculate the activity of a species.

        Args:
        -----
        - mass_concentration (float): Concentration of the species [kg/m^3]

        Returns:
        --------
        - float: Activity of the particle [unitless].
        """
        pass