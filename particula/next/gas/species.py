"""
Gas Species module.

Units are in kg/mol for molar mass, Kelvin for temperature, Pascals for
pressure, and kg/m^3 for concentration.
"""


from typing import Union
from numpy.typing import NDArray
import numpy as np
from particula.next.gas.vapor_pressure import VaporPressureStrategy


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
        return str(self.name)

    def __len__(self):
        """Return the number of gas species."""
        return len(self.molar_mass)

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

    def set_vapor_pressure_strategy(
            self,
            strategy: Union[VaporPressureStrategy, list[VaporPressureStrategy]]
            ):
        """Set the vapor pressure strategies for the gas species.

        Args:
        - strategy (VaporPressureStrategy): The strategy for calculating the
            pure vapor pressure of the gas species. either a single strategy or
            a list of strategies."""
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

    def get_concentration(self) -> Union[float, NDArray[np.float_]]:
        """Get the concentration of the gas species in the mixture, in kg/m^3.

        Returns:
        - concentration (float or NDArray[np.float_]): The concentration of the
            gas species in the mixture."""
        if self.concentration is None:
            raise ValueError("Concentration property is not set.")
        return self.concentration

    def get_pure_vapor_pressure(
        self,
        temperature: Union[float, NDArray[np.float_]]
    ) -> Union[float, NDArray[np.float_]]:
        """Calculate the pure vapor pressure of the gas species at a given
        temperature in Kelvin.

        This method supports both a single strategy or a list of strategies
        for calculating vapor pressure.

        Args:
        - temperature (float or NDArray[np.float_]): The temperature in
        Kelvin at which to calculate vapor pressure.

        Returns:
        - vapor_pressure (float or NDArray[np.float_]): The calculated pure
        vapor pressure in Pascals.

        Raises:
            ValueError: If no vapor pressure strategy is set.
        """
        if self.pure_vapor_pressure_strategy is None:
            raise ValueError("Vapor pressure strategy is not set.")

        if isinstance(self.pure_vapor_pressure_strategy, list):
            # Handle a list of strategies: calculate and return a list of vapor
            # pressures
            return np.array(
                [strategy.pure_vapor_pressure(temperature)
                 for strategy in self.pure_vapor_pressure_strategy],
                dtype=np.float_)

        # Handle a single strategy: calculate and return the vapor pressure
        return self.pure_vapor_pressure_strategy.pure_vapor_pressure(
            temperature)

    def get_partial_pressure(
        self,
        temperature: Union[float, NDArray[np.float_]]
    ) -> Union[float, NDArray[np.float_]]:
        """
        Calculate the partial pressure of the gas based on the vapor
        pressure strategy. This method accounts for multiple strategies if
        assigned and calculates partial pressure for each strategy based on
        the corresponding concentration and molar mass.

        Parameters:
        - temperature (float or NDArray[np.float_]): The temperature in
        Kelvin at which to calculate the partial pressure.

        Returns:
        - partial_pressure (float or NDArray[np.float_]): Partial pressure
        of the gas in Pascals.

        Raises:
        - ValueError: If the vapor pressure strategy is not set.
        """
        # Check if the vapor pressure strategy is set
        if self.pure_vapor_pressure_strategy is None:
            raise ValueError("Vapor pressure strategy is not set.")

        # Handle multiple vapor pressure strategies
        if isinstance(self.pure_vapor_pressure_strategy, list):
            # Calculate partial pressure for each strategy
            return np.array(
                [strategy.partial_pressure(
                    concentration=c,
                    molar_mass=m,
                    temperature=temperature)
                 for (strategy, c, m) in zip(
                     self.pure_vapor_pressure_strategy,
                     self.concentration,
                     self.molar_mass)],
                dtype=np.float_
            )
        # Calculate partial pressure using a single strategy
        return self.pure_vapor_pressure_strategy.partial_pressure(
            concentration=self.concentration,
            molar_mass=self.molar_mass,
            temperature=temperature
        )

    def get_saturation_ratio(
        self,
        temperature: Union[float, NDArray[np.float_]]
    ) -> Union[float, NDArray[np.float_]]:
        """
        Calculate the saturation ratio of the gas based on the vapor
        pressure strategy. This method accounts for multiple strategies if
        assigned and calculates saturation ratio for each strategy based on
        the corresponding concentration and molar mass.

        Parameters:
        - temperature (float or NDArray[np.float_]): The temperature in
        Kelvin at which to calculate the partial pressure.

        Returns:
        - saturation_ratio (float or NDArray[np.float_]): The saturation ratio
        of the gas

        Raises:
        - ValueError: If the vapor pressure strategy is not set.
        """
        # Check if the vapor pressure strategy is set
        if self.pure_vapor_pressure_strategy is None:
            raise ValueError("Vapor pressure strategy is not set.")

        # Handle multiple vapor pressure strategies
        if isinstance(self.pure_vapor_pressure_strategy, list):
            # Calculate saturation ratio for each strategy
            return np.array(
                [strategy.saturation_ratio(
                    concentration=c,
                    molar_mass=m,
                    temperature=temperature)
                 for (strategy, c, m) in zip(
                     self.pure_vapor_pressure_strategy,
                     self.concentration,
                     self.molar_mass)],
                dtype=np.float_
            )
        # Calculate saturation ratio using a single strategy
        return self.pure_vapor_pressure_strategy.saturation_ratio(
            concentration=self.concentration,
            molar_mass=self.molar_mass,
            temperature=temperature
        )

    def get_saturation_concentration(
        self,
        temperature: Union[float, NDArray[np.float_]]
    ) -> Union[float, NDArray[np.float_]]:
        """
        Calculate the saturation concentration of the gas based on the vapor
        pressure strategy. This method accounts for multiple strategies if
        assigned and calculates saturation concentration for each strategy
        based on the molar mass.

        Parameters:
        - temperature (float or NDArray[np.float_]): The temperature in
        Kelvin at which to calculate the partial pressure.

        Returns:
        - saturation_concentration (float or NDArray[np.float_]): The
        saturation concentration of the gas

        Raises:
        - ValueError: If the vapor pressure strategy is not set.
        """
        # Check if the vapor pressure strategy is set
        if self.pure_vapor_pressure_strategy is None:
            raise ValueError("Vapor pressure strategy is not set.")

        # Handle multiple vapor pressure strategies
        if isinstance(self.pure_vapor_pressure_strategy, list):
            # Calculate saturation concentraiton for each strategy
            return np.array(
                [strategy.saturation_concentration(
                    molar_mass=m,
                    temperature=temperature)
                 for (strategy, m) in zip(
                     self.pure_vapor_pressure_strategy,
                     self.molar_mass)],
                dtype=np.float_
            )
        # Calculate saturation concentration using a single strategy
        return self.pure_vapor_pressure_strategy.saturation_concentration(
            molar_mass=self.molar_mass,
            temperature=temperature
        )

    def add_concentration(
        self,
        added_concentration: Union[float, NDArray[np.float_]]
    ):
        """Add concentration to the gas species.

        Args:
        - added_concentration (float): The concentration to add to the gas
            species."""
        self.concentration = self.concentration + added_concentration


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

    def vapor_pressure_strategy(
            self,
            strategy: Union[VaporPressureStrategy, list[VaporPressureStrategy]]
            ):
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
