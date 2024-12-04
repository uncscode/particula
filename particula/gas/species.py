"""
Gas Species module.

Units are in kg/mol for molar mass, Kelvin for temperature, Pascals for
pressure, and kg/m^3 for concentration.
"""

import logging
import warnings
from typing import Union
from numpy.typing import NDArray
import numpy as np
from particula.gas.vapor_pressure_strategies import (
    VaporPressureStrategy,
    ConstantVaporPressureStrategy,
)

logger = logging.getLogger("particula")


class GasSpecies:
    """GasSpecies represents an individual or array of gas species with
    properties like name, molar mass, vapor pressure, and condensability.

    Attributes:
        - name : The name of the gas species.
        - molar_mass : The molar mass of the gas species.
        - pure_vapor_pressure_strategy : The strategy for calculating the pure
            vapor pressure of the gas species. Can be a single strategy or a
            list of strategies. Default is a constant vapor pressure strategy
            with a vapor pressure of 0.0 Pa.
        - condensable : Indicates whether the gas species is condensable.
            Default is True.
        - concentration : The concentration of the gas species in the mixture.
            Default is 0.0 kg/m^3.
    """

    def __init__(  # pylint: disable=too-many-positional-arguments
        # pylint: disable=too-many-arguments
        self,
        name: Union[str, NDArray[np.str_]],
        molar_mass: Union[float, NDArray[np.float64]],
        vapor_pressure_strategy: Union[
            VaporPressureStrategy, list[VaporPressureStrategy]
        ] = ConstantVaporPressureStrategy(0.0),
        condensable: Union[bool, NDArray[np.bool_]] = True,
        concentration: Union[float, NDArray[np.float64]] = 0.0,
    ) -> None:
        self._check_non_positive_value(molar_mass, "molar mass")
        self.molar_mass = molar_mass
        concentration = self._check_if_negative_concentration(concentration)
        self.concentration = concentration

        self.name = name
        self.pure_vapor_pressure_strategy = vapor_pressure_strategy
        self.condensable = condensable

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
            The name of the gas species.
        """
        return self.name

    def get_molar_mass(self) -> Union[float, NDArray[np.float64]]:
        """Get the molar mass of the gas species in kg/mol.

        Returns:
            The molar mass of the gas species, in kg/mol.
        """
        return self.molar_mass

    def get_condensable(self) -> Union[bool, NDArray[np.bool_]]:
        """Check if the gas species is condensable or not.

        Returns:
            True if the gas species is condensable, False otherwise.
        """
        return self.condensable

    def get_concentration(self) -> Union[float, NDArray[np.float64]]:
        """Get the concentration of the gas species in the mixture, in kg/m^3.

        Returns:
            The concentration of the gas species in the mixture.
        """
        return self.concentration

    def get_pure_vapor_pressure(
        self, temperature: Union[float, NDArray[np.float64]]
    ) -> Union[float, NDArray[np.float64]]:
        """Calculate the pure vapor pressure of the gas species at a given
        temperature in Kelvin.

        This method supports both a single strategy or a list of strategies
        for calculating vapor pressure.

        Args:
            - temperature : The temperature in Kelvin at which to calculate
                vapor pressure.

        Returns:
            The calculated pure vapor pressure in Pascals.

        Raises:
            ValueError: If no vapor pressure strategy is set.

        Example:
            ``` py title="Example usage of get_pure_vapor_pressure"
            gas_object.get_pure_vapor_pressure(temperature=298)
            ```
        """
        if isinstance(self.pure_vapor_pressure_strategy, list):
            # Handle a list of strategies: calculate and return a list of vapor
            # pressures
            return np.array(
                [
                    strategy.pure_vapor_pressure(temperature)
                    for strategy in self.pure_vapor_pressure_strategy
                ],
                dtype=np.float64,
            )

        # Handle a single strategy: calculate and return the vapor pressure
        return self.pure_vapor_pressure_strategy.pure_vapor_pressure(
            temperature
        )

    def get_partial_pressure(
        self, temperature: Union[float, NDArray[np.float64]]
    ) -> Union[float, NDArray[np.float64]]:
        """Calculate the partial pressure of the gas based on the vapor
        pressure strategy.

        This method accounts for multiple strategies if assigned and
        calculates partial pressure for each strategy based on the
        corresponding concentration and molar mass.

        Args:
            - temperature : The temperature in Kelvin at which to calculate
                the partial pressure.

        Returns:
            Partial pressure of the gas in Pascals.

        Raises:
            ValueError: If the vapor pressure strategy is not set.

        Example:
            ``` py title="Example usage of get_partial_pressure"
            gas_object.get_partial_pressure(temperature=298)
            ```
        """
        # Handle multiple vapor pressure strategies
        if isinstance(self.pure_vapor_pressure_strategy, list):
            # Calculate partial pressure for each strategy
            return np.array(
                [
                    strategy.partial_pressure(
                        concentration=c, molar_mass=m, temperature=temperature
                    )
                    for (strategy, c, m) in zip(
                        self.pure_vapor_pressure_strategy,
                        self.concentration,  # type: ignore
                        self.molar_mass,  # type: ignore
                    )
                ],
                dtype=np.float64,
            )
        # Calculate partial pressure using a single strategy
        return self.pure_vapor_pressure_strategy.partial_pressure(
            concentration=self.concentration,
            molar_mass=self.molar_mass,
            temperature=temperature,
        )

    def get_saturation_ratio(
        self, temperature: Union[float, NDArray[np.float64]]
    ) -> Union[float, NDArray[np.float64]]:
        """Calculate the saturation ratio of the gas based on the vapor
        pressure strategy.

        This method accounts for multiple strategies if assigned and
        calculates saturation ratio for each strategy based on the
        corresponding concentration and molar mass.

        Args:
            - temperature : The temperature in Kelvin at which to calculate
                the partial pressure.

        Returns:
            The saturation ratio of the gas.

        Raises:
            ValueError : If the vapor pressure strategy is not set.

        Example:
            ``` py title="Example usage of get_saturation_ratio"
            gas_object.get_saturation_ratio(temperature=298)
            ```
        """
        # Handle multiple vapor pressure strategies
        if isinstance(self.pure_vapor_pressure_strategy, list):
            # Calculate saturation ratio for each strategy
            return np.array(
                [
                    strategy.saturation_ratio(
                        concentration=c, molar_mass=m, temperature=temperature
                    )
                    for (strategy, c, m) in zip(
                        self.pure_vapor_pressure_strategy,
                        self.concentration,  # type: ignore
                        self.molar_mass,  # type: ignore
                    )
                ],
                dtype=np.float64,
            )
        # Calculate saturation ratio using a single strategy
        return self.pure_vapor_pressure_strategy.saturation_ratio(
            concentration=self.concentration,
            molar_mass=self.molar_mass,
            temperature=temperature,
        )

    def get_saturation_concentration(
        self, temperature: Union[float, NDArray[np.float64]]
    ) -> Union[float, NDArray[np.float64]]:
        """Calculate the saturation concentration of the gas based on the
        vapor pressure strategy.

        This method accounts for multiple strategies if assigned and
        calculates saturation concentration for each strategy based on the
        molar mass.

        Args:
            - temperature : The temperature in Kelvin at which to calculate
                the partial pressure.

        Returns:
            The saturation concentration of the gas.

        Raises:
            ValueError: If the vapor pressure strategy is not set.

        Example:
            ``` py title="Example usage of get_saturation_concentration"
            gas_object.get_saturation_concentration(temperature=298)
            ```
        """
        # Handle multiple vapor pressure strategies
        if isinstance(self.pure_vapor_pressure_strategy, list):
            # Calculate saturation concentration for each strategy
            return np.array(
                [
                    strategy.saturation_concentration(
                        molar_mass=m, temperature=temperature
                    )
                    for (strategy, m) in zip(
                        self.pure_vapor_pressure_strategy,
                        self.molar_mass,  # type: ignore
                    )
                ],
                dtype=np.float64,
            )
        # Calculate saturation concentration using a single strategy
        return self.pure_vapor_pressure_strategy.saturation_concentration(
            molar_mass=self.molar_mass, temperature=temperature
        )

    def add_concentration(
        self, added_concentration: Union[float, NDArray[np.float64]]
    ) -> None:
        """Add concentration to the gas species.

        Args:
            - added_concentration : The concentration to add to the gas
                species.

        Example:
            ``` py title="Example usage of add_concentration"
            gas_object.add_concentration(added_concentration=1e-10)
            ```
        """
        self.set_concentration(self.concentration + added_concentration)

    def set_concentration(
        self, new_concentration: Union[float, NDArray[np.float64]]
    ) -> None:
        """Set the concentration of the gas species.

        Args:
            - new_concentration : The new concentration of the gas species.

        Example:
            ``` py title="Example usage of set_concentration"
            gas_object.set_concentration(new_concentration=1e-10)
            ```
        """
        new_concentration = self._check_if_negative_concentration(
            new_concentration)
        self.concentration = new_concentration

    def _check_if_negative_concentration(
        self, values: Union[float, NDArray[np.float64]]
    ) -> None:
        """Log a warning if the concentration is negative."""
        if np.any(values < 0.0):
            message = "Negative concentration in gas species, set = 0."
            logger.warning(message)
            warnings.warn(message, UserWarning)
            # Set negative concentrations to 0
            values = np.maximum(values, 0.0)
        return values

    def _check_non_positive_value(
        self, value: Union[float, NDArray[np.float64]], name: str
    ) -> None:
        """Check for non-positive values and raise an error if found."""
        if np.any(value <= 0.0):
            message = f"Non-positive {name} in gas species, stopping."
            logger.error(message)
            raise ValueError(message)
