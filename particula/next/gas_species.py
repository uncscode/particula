"""
Gas Species module.

Units are in kg/mol for molar mass, Kelvin for temperature, Pascals for
pressure, and kg/m^3 for concentration.
"""


from typing import Union
from numpy.typing import NDArray
import numpy as np
from particula.next.vapor_pressure import VaporPressureStrategy


class GasSpecies:
    """GasSpecies represents an individual or array of gas species with
    properties like name, molar mass, vapor pressure, and condensability.

    Attributes:
    - name (str): The name of the gas species.
    - molar_mass (float): The molar mass of the gas species.
    - pure_vapor_pressure_strategy (VaporPressureStrategy): The strategy for
        calculating the pure vapor pressure of the gas species.
    - condensable (bool): Indicates whether the gas species is condensable.
    """

    def __init__(self):
        self.name = None
        self.molar_mass = None
        self.pure_vapor_pressure_strategy = None
        self.condensable = None
        self.concentration = None

    def __str__(self):
        """Return a string representation of the GasSpecies object."""
        return f"GasSpecies(name={self.name})"

    def set_name(self, name: Union[str, NDArray[np.str_]]):
        """Set the name of the gas species.

        Args:
        - name (str or NDArray[np.str_]): The name of the gas species."""
        self.name = name
        return self

    def set_molar_mass(self, molar_mass: Union[float, NDArray[np.float_]]):
        """Set the molar mass of the gas species in kg/mol.

        Args:
        - molar_mass (float or NDArray[np.float_]): The molar mass of the gas
            species in kg/mol."""
        self.molar_mass = molar_mass
        return self

    def set_vapor_pressure_strategy(self, strategy: VaporPressureStrategy):
        """Set the vapor pressure strategy for the gas species.

        Args:
        - strategy (VaporPressureStrategy): The strategy for calculating the
            pure vapor pressure of the gas species."""
        self.pure_vapor_pressure_strategy = strategy
        return self

    def set_condensable(self, condensable: Union[bool, NDArray[np.bool_]]):
        """Set the condensable bool of the gas species."""
        self.condensable = condensable
        return self

    def set_concentration(
            self, concentration: Union[float, NDArray[np.float_]]):
        """Set the concentration of the gas species in the mixture, in kg/m^3.

        Args:
        - concentration (float or NDArray[np.float_]): The concentration of the
            gas species in the mixture."""
        self.concentration = concentration
        return self

    def get_molar_mass(self) -> Union[float, NDArray[np.float_]]:
        """Get the molar mass of the gas species in kg/mol.

        Returns:
        - molar_mass (float or NDArray[np.float_]): The molar mass of the gas
            species, in kg/mol."""
        if self.molar_mass is None:
            raise ValueError("Molar mass property is not set.")
        return self.molar_mass

    def get_condensable(self) -> Union[bool, NDArray[np.bool_]]:
        """Check if the gas species is condensable or not.

        Returns:
        - condensable (bool): True if the gas species is condensable, False
            otherwise."""
        if self.condensable is None:
            raise ValueError("Condensable property is not set.")
        return self.condensable

    def get_pure_vapor_pressure(
        self,
        temperature: Union[float, NDArray[np.float_]]
    ) -> Union[float, NDArray[np.float_]]:
        """Calculate the pure vapor pressure of the gas species at a given
        temperature in Kelvin.

        Args:
        - temperature (float or NDArray[np.float_]): The temperature in Kelvin.

        Returns:
        - vapor_pressure (float or NDArray[np.float_]): The pure vapor pressure
            in Pascals."""
        if not self.pure_vapor_pressure_strategy:
            raise ValueError("Vapor pressure strategy is not set.")
        return self.pure_vapor_pressure_strategy.pure_vapor_pressure(
            temperature)

    def get_concentration(self) -> Union[float, NDArray[np.float_]]:
        """Get the concentration of the gas species in the mixture, in kg/m^3.

        Returns:
        - concentration (float or NDArray[np.float_]): The concentration of the
            gas species in the mixture."""
        if self.concentration is None:
            raise ValueError("Concentration property is not set.")
        return self.concentration


class GasSpeciesBuilder:
    """Builder class for GasSpecies objects, allowing for a more fluent and
    readable creation of GasSpecies instances with optional parameters.

    Methods:
    - name: Set the name of the gas species.
    - molar_mass: Set the molar mass of the gas species.
    - vapor_pressure_strategy: Set the vapor pressure strategy for the gas
        species.
    - condensable: Set the condensable property of the gas species.
    - build: Validate and return the GasSpecies object.

    Returns:
    - GasSpecies: The built GasSpecies object.
    """

    def __init__(self):
        self.gas_species = GasSpecies()

    def name(self, name: Union[str, NDArray[np.str_]]):
        """Set the name of the gas species."""
        self.gas_species.set_name(name)
        return self

    def molar_mass(self, molar_mass: Union[float, NDArray[np.float_]]):
        """Set the molar mass of the gas species. Units in kg/mol."""
        self.gas_species.set_molar_mass(molar_mass)
        return self

    def vapor_pressure_strategy(self, strategy: VaporPressureStrategy):
        """Set the vapor pressure strategy for the gas species."""
        self.gas_species.set_vapor_pressure_strategy(strategy)
        return self

    def condensable(self, condensable: Union[bool, NDArray[np.bool_]]):
        """Set the condensable bool of the gas species."""
        self.gas_species.set_condensable(condensable)
        return self

    def concentration(self, concentration: Union[float, NDArray[np.float_]]):
        """Set the concentration of the gas species in the mixture,
        in kg/m^3."""
        self.gas_species.set_concentration(concentration)
        return self

    def build(self) -> GasSpecies:
        """Validate and return the GasSpecies object."""
        if self.gas_species.name is None:
            raise ValueError("Gas species name is required.")
        if self.gas_species.molar_mass is None:
            raise ValueError("Gas species molar mass is required")
        if self.gas_species.pure_vapor_pressure_strategy is None:
            raise ValueError("Vapor pressure strategy is required")
        if self.gas_species.condensable is None:
            raise ValueError("Condensable property is required")
        if self.gas_species.concentration is None:
            raise ValueError("Concentration property is required")
        return self.gas_species
