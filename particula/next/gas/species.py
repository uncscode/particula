"""
This module demonstrates the Composite Pattern through the modeling of gases
and their constituent species. In this pattern, both composite objects (gases)
and leaf objects (gas species) are treated uniformly.

The Composite Pattern is particularly useful in scenarios where a part-whole
hierarchy is present. In our case, a `Gas` represents the whole, which can be
composed of various parts - the `GasSpecies`. Each `GasSpecies` can have
distinct properties such as mass and vapor pressure, and the `Gas` itself can
consist of multiple species, mimicking real-world gas mixtures.

Classes:
- GasSpecies: Represents an individual gas species with properties like name,
mass, and vapor pressure. Acts as a leaf in the composite structure.
- Gas: Represents a composite object that can contain multiple GasSpecies.
It manages the collection of these species and performs operations on them as
a group.

The Composite Pattern allows for the Gas class to manage an arbitrary number
of GasSpecies objects, facilitating operations that need to consider the gas
mixture as a whole or interact with its individual components seamlessly.
"""

from typing import Optional
from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray
import numpy as np
from particula.next.gas.vapor_pressure_strategies import (
    VaporPressureStrategy, ConstantVaporPressureStrategy)


@dataclass
class GasSpecies:
    """
    Represents a single species of gas, including its properties such as
    name, mass, vapor pressure, and whether it is condensable.

    Attributes:
    - name (str): The name of the gas species.
    - mass (float): The mass of the gas species.
    - vapor_pressure (Optional[float]): The vapor pressure of the gas
        species. None if not applicable.
    - condensable (bool): Indicates whether the gas species is condensable.
        Default is True.
    - concentration (float): The concentration of the gas species in the
        mixture. Default is 0.0 kg/m^3.

    Methods:
    --------
    - get_molar_mass: Get the molar mass of the gas species.
    - get_condensable: Check if the gas species is condensable.
    - get_concentration: Get the concentration of the gas species in the
        mixture.
    - get_pure_vapor_pressure: Calculate the pure vapor pressure of the gas
        species at a given temperature.
    - get_partial_pressure: Calculate the partial pressure of the gas species.
    - get_saturation_ratio: Calculate the saturation ratio of the gas species.
    - get_saturation_concentration: Calculate the saturation concentration of
        the gas species.
    - add_concentration: Add concentration to the gas species.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        name: Union[str, NDArray[np.str_]],
        molar_mass: Union[float, NDArray[np.float_]],
        vapor_pressure_strategy: Union[
            VaporPressureStrategy, list[VaporPressureStrategy]
            ] = ConstantVaporPressureStrategy(0.0),
        condensable: Union[bool, NDArray[np.bool_]] = True,
        concentration: Union[float, NDArray[np.float_]] = 0.0
    ) -> None:
        self.name = name
        self.molar_mass = molar_mass
        self.pure_vapor_pressure_strategy = vapor_pressure_strategy
        self.condensable = condensable
        self.concentration = concentration

    def __str__(self):
        """Return a string representation of the GasSpecies object."""
        return str(self.name)

    def __len__(self):
        """Return the number of gas species."""
        return (
            len(self.molar_mass)
            if isinstance(self.molar_mass, np.ndarray)
            else 1.0
        )

    def get_name(self) -> Union[str, NDArray[np.str_]]:
        """Get the name of the gas species.

        Returns:
            np.float64: The mass of the gas species.
        """
        return np.float64(self.mass)

    def get_molar_mass(self) -> Union[float, NDArray[np.float_]]:
        """Get the molar mass of the gas species in kg/mol.

        Returns:
        - molar_mass (float or NDArray[np.float_]): The molar mass of the gas
            species, in kg/mol."""
        return self.molar_mass

    def get_condensable(self) -> Union[bool, NDArray[np.bool_]]:
        """Check if the gas species is condensable or not.

        Returns:
        - condensable (bool): True if the gas species is condensable, False
            otherwise."""
        return self.condensable

    def get_concentration(self) -> Union[float, NDArray[np.float_]]:
        """Get the concentration of the gas species in the mixture, in kg/m^3.

        Returns:
        - concentration (float or NDArray[np.float_]): The concentration of the
            gas species in the mixture."""
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
            ValueError: If the specified name is not found in the mixture.
        """
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
        Returns the mass of a specific condensable species or the masses of
        all condensable species in the gas mixture as an np.ndarray. If a name
        is provided, only the mass of that specific condensable species is
        returned.

        Parameters:
        - temperature (float or NDArray[np.float_]): The temperature in
        Kelvin at which to calculate the partial pressure.

        Returns:
        - saturation_concentration (float or NDArray[np.float_]): The
        saturation concentration of the gas

        Raises:
            ValueError: If a specific species name is provided but not found
            in the mixture, or if it's not condensable.
        """
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
