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
from particula.util.validate_inputs import validate_inputs


logger = logging.getLogger("particula")


class GasSpecies:
    """
    Represents an individual or array of gas species with properties like
    name, molar mass, vapor pressure, and condensability.

    Attributes:
        - name : The name of the gas species.
        - molar_mass : The molar mass of the gas species in kg/mol.
        - pure_vapor_pressure_strategy : The strategy (or list of strategies)
          for calculating the pure vapor pressure of the gas species.
        - condensable : Indicates whether the gas species is condensable.
        - concentration : The concentration of the gas species in kg/m^3.

    Methods:
    - get_name : Return the name of the gas species.
    - get_molar_mass : Return the molar mass in kg/mol.
    - get_condensable : Return whether the species is condensable.
    - get_concentration : Return the concentration in kg/m^3.
    - get_pure_vapor_pressure : Calculate pure vapor pressure at a given Temp.
    - get_partial_pressure : Calculate partial pressure at a given Temp.
    - get_saturation_ratio : Calculate saturation ratio at a given Temp.
    - get_saturation_concentration : Calculate saturation concentration at a
      given Temperature.
    - add_concentration : Add concentration to the species.
    - set_concentration : Overwrite concentration value.

    Examples:
        ```py title="GasSpecies usage example"
        import particula as par
        constant_vapor_pressure = par.gas.ConstantVaporPressureStrategy(2330)
        species = par.gas.GasSpecies(
            name="Water",
            molar_mass=0.018,
            vapor_pressure_strategy=constant_vapor_pressure,
            condensable=True,
            concentration=1e-3,  # kg/m^3
        )
        print(species.get_name(), species.get_concentration())
        ```
    """

    @validate_inputs({"molar_mass": "positive"})
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
        """
        Initialize the GasSpecies with name, molar mass, and vapor pressure
        strategy.

        Arguments:
            - name : The name of the gas species.
            - molar_mass : The molar mass in kg/mol (must be > 0).
            - vapor_pressure_strategy : A single or list of strategies for
              calculating vapor pressure.
            - condensable : Whether the species is condensable.
            - concentration : The initial concentration in kg/m^3.

        Raises:
            - ValueError : If molar_mass is non-positive.
        """
        self.molar_mass = molar_mass
        concentration = self._check_if_negative_concentration(concentration)
        self.concentration = concentration

        self.name = name
        self.pure_vapor_pressure_strategy = vapor_pressure_strategy
        self.condensable = condensable

    def __str__(self):
        """
        Return a string representation of the GasSpecies object.

        Returns:
            - str : The string name of the gas species.
        """
        return str(self.name)

    def __len__(self):
        """
        Return the number of gas species (1 if scalar; array length if
        ndarray).

        Returns:
            - float or int : Number of species (array length or 1).

        Examples:
            ```py title="Example of len()"
            len(gas_object)
            ```
        """
        return (
            len(self.molar_mass)
            if isinstance(self.molar_mass, np.ndarray)
            else 1.0
        )

    def get_name(self) -> Union[str, NDArray[np.str_]]:
        """
        Return the name of the gas species.

        Returns:
            - Name of the gas species.

        Examples:
            ```py title="Example of get_name()"
            gas_object.get_name()
            ```
        """
        return self.name

    def get_molar_mass(self) -> Union[float, NDArray[np.float64]]:
        """
        Return the molar mass of the gas species in kg/mol.

        Returns:
            - Molar mass in kg/mol.

        Examples:
            ```py title="Example of get_molar_mass()"
            gas_object.get_molar_mass()
            ```
        """
        return self.molar_mass

    def get_condensable(self) -> Union[bool, NDArray[np.bool_]]:
        """
        Check if the gas species is condensable.

        Returns:
            - True if condensable, else False.

        Examples:
            ``` py title="Example of get_condensable()"
            gas_object.get_condensable()
            ```
        """
        return self.condensable

    def get_concentration(self) -> Union[float, NDArray[np.float64]]:
        """
        Return the concentration of the gas species in kg/m^3.

        Returns:
            - Species concentration.
        """
        return self.concentration

    def get_pure_vapor_pressure(
        self, temperature: Union[float, NDArray[np.float64]]
    ) -> Union[float, NDArray[np.float64]]:
        """
        Calculate the pure vapor pressure at a given temperature (K).

        Arguments:
            - temperature : The temperature in Kelvin.

        Returns:
            - Pure vapor pressure in Pa.

        Raises:
            - ValueError : If no vapor pressure strategy is set.

        Examples:
            ```py title="Example"
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
        """
        Calculate the partial pressure of the gas at a given temperature (K).

        Arguments:
            - temperature : The temperature in Kelvin.

        Returns:
            - Partial pressure in Pa.

        Raises:
            - ValueError : If the vapor pressure strategy is not set.

        Examples:
            ```py title="Example of get_partial_pressure()"
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
        """
        Calculate the saturation ratio of the gas at a given temperature (K).

        Arguments:
            - temperature : The temperature in Kelvin.

        Returns:
            - The saturation ratio.

        Raises:
            - ValueError : If the vapor pressure strategy is not set.

        Examples:
            ```py title="Example of get_saturation_ratio()"
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
        """
        Calculate the saturation concentration at a given temperature (K).

        Arguments:
            - temperature : The temperature in Kelvin.

        Returns:
            - The saturation concentration in kg/m^3.

        Raises:
            - ValueError : If the vapor pressure strategy is not set.

        Examples:
            ```py title="Example of get_saturation_concentration()"
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
        """
        Add concentration (kg/m^3) to the gas species.

        Arguments:
            - added_concentration : The amount to add in kg/m^3.

        Examples:
            ```py title="Example of add_concentration()"
            gas_object.add_concentration(1e-10)
            ```
        """
        self.set_concentration(self.concentration + added_concentration)

    def set_concentration(
        self, new_concentration: Union[float, NDArray[np.float64]]
    ) -> None:
        """
        Overwrite the concentration of the gas species in kg/m^3.

        Arguments:
            - new_concentration : The new concentration value in kg/m^3.

        Examples:
            ```py title="Example of set_concentration()"
            gas_object.set_concentration(1e-10)
            ```
        """
        new_concentration = self._check_if_negative_concentration(
            new_concentration
        )
        self.concentration = new_concentration

    def _check_if_negative_concentration(
        self, values: Union[float, NDArray[np.float64]]
    ) -> Union[float, NDArray[np.float64]]:
        """
        Ensure concentration is not negative. Log a warning if it is and set
        to 0.

        Arguments:
            - values : Concentration values to check.

        Returns:
            - Corrected concentration (≥ 0).
        """
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
        """
        Raise an error if any value is non-positive.

        Arguments:
            - value : The numeric value(s) to check.
            - name : Name of the parameter for the error message.

        Raises:
            - ValueError : If any value <= 0 is detected.
        """
        if np.any(value <= 0.0):
            message = f"Non-positive {name} in gas species, stopping."
            logger.error(message)
            raise ValueError(message)
