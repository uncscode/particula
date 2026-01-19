"""Gas Species module.

Units are in kg/mol for molar mass, Kelvin for temperature, Pascals for
pressure, and kg/m^3 for concentration.
"""

import copy
import logging
import warnings
from typing import Union

import numpy as np
from numpy.typing import NDArray

from particula.gas.vapor_pressure_strategies import (
    ConstantVaporPressureStrategy,
    VaporPressureStrategy,
)
from particula.util.validate_inputs import validate_inputs

logger = logging.getLogger("particula")


class GasSpecies:
    """Represents an individual or array of gas species with properties.

    Attributes:
        - name : The name of the gas species.
        - molar_mass : The molar mass of the gas species in kg/mol.
        - pure_vapor_pressure_strategy : The strategy (or list of strategies)
          for calculating the pure vapor pressure of the gas species.
        - partitioning : Indicates whether the gas species can partition.
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
    - append : Append another GasSpecies instance to this one.
    - __iadd__ : In-place addition of another GasSpecies instance.
    - __add__ : Addition of two GasSpecies instances (non-mutating).
    - __str__ : String representation of the GasSpecies object.
    - __len__ : Number of gas species (1 if scalar; array length if ndarray).

    Examples:
        ```py title="GasSpecies usage example"
        import particula as par
        constant_vapor_pressure = par.gas.ConstantVaporPressureStrategy(2330)
        species = par.gas.GasSpecies(
            name="Water",
            molar_mass=0.018,
            vapor_pressure_strategy=constant_vapor_pressure,
            partitioning=True,
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
        ] = ConstantVaporPressureStrategy(0.0),  # noqa: B008
        partitioning: bool = True,
        concentration: Union[float, NDArray[np.float64]] = 0.0,
    ) -> None:
        """Initialize the with name, molar mass, and vapor pressure strategy.

        Arguments:
            - name : The name of the gas species.
            - molar_mass : The molar mass in kg/mol (must be > 0).
            - vapor_pressure_strategy : A single or list of strategies for
              calculating vapor pressure.
            - partitioning : Whether the species can partition.
            - concentration : The initial concentration in kg/m^3.

        Raises:
            - ValueError : If molar_mass is non-positive.
        """
        self.molar_mass = molar_mass
        concentration = self._check_if_negative_concentration(concentration)
        self.concentration = concentration

        self.name = name
        self.pure_vapor_pressure_strategy = vapor_pressure_strategy
        self.partitioning = partitioning

    def __str__(self):
        """Return a string representation of the GasSpecies object.

        Returns:
            - str : The string name of the gas species.
        """
        return str(self.name)

    def __len__(self):
        """Return the number of gas species.

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
            else 1
        )

    def __iadd__(self, other: "GasSpecies") -> "GasSpecies":
        """In-place addition: append another GasSpecies object to this one.

        Arguments:
            - other : The GasSpecies instance whose attributes will be
              appended to the current object.

        Returns:
            - GasSpecies : The mutated object (`self`) containing the combined
              attributes.

        Raises:
            - TypeError : If *other* is not a GasSpecies instance.
            - ValueError : If the two objects have different ``partitioning``
              flags.

        Examples:
            ```py title="Using the += operator"
            species1 += species2
            ```
        """
        self.append(other)
        return self

    def __add__(self, other: "GasSpecies") -> "GasSpecies":
        """Addition of two GasSpecies objects (non-mutating).

        Creates and returns a new GasSpecies instance that contains the
        combined attributes of *self* and *other*.

        Arguments:
            - other : The GasSpecies instance to be combined with *self*.

        Returns:
            - GasSpecies : A new object with concatenated attributes.

        Raises:
            - TypeError : If *other* is not a GasSpecies instance.
            - ValueError : If the two objects have different ``partitioning``
              flags.

        Examples:
            ```py title="Using the + operator"
            merged_species = species1 + species2
            ```
        """
        new_species = copy.deepcopy(self)
        new_species.append(other)
        return new_species

    def get_name(self) -> Union[str, NDArray[np.str_]]:
        """Return the name of the gas species.

        Returns:
            - Name of the gas species.

        Examples:
            ```py title="Example of get_name()"
            gas_object.get_name()
            ```
        """
        return self.name

    def get_molar_mass(self) -> Union[float, NDArray[np.float64]]:
        """Return the molar mass of the gas species in kg/mol.

        Returns:
            - Molar mass in kg/mol.

        Examples:
            ```py title="Example of get_molar_mass()"
            gas_object.get_molar_mass()
            ```
        """
        return self.molar_mass

    def get_partitioning(self) -> bool:
        """Return the partitioning flag (True if the species can partition)."""
        return self.partitioning

    def get_concentration(self) -> Union[float, NDArray[np.float64]]:
        """Return the concentration of the gas species in kg/m^3.

        Returns:
            - Species concentration.
        """
        return self.concentration

    def get_pure_vapor_pressure(
        self, temperature: Union[float, NDArray[np.float64]]
    ) -> Union[float, NDArray[np.float64]]:
        """Calculate the pure vapor pressure at a given temperature (K).

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
        """Calculate the partial pressure of the gas at a given temperature (K).

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
                        strict=True,
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
        """Calculate the saturation ratio of the gas at a given temperature (K).

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
                        strict=True,
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
        """Calculate the saturation concentration at a given temperature (K).

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
                        strict=True,
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
        """Add concentration (kg/m^3) to the gas species.

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
        """Overwrite the concentration of the gas species in kg/m^3.

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

    def append(self, other: "GasSpecies") -> None:
        """Append another GasSpecies instance to this one (in-place).

        Arguments:
            - other : The GasSpecies object whose attributes will be
              concatenated with those of the current object.

        Returns:
            - None : The method mutates ``self`` and returns ``None``.

        Raises:
            - TypeError  : If *other* is not a GasSpecies instance.
            - ValueError : If *other* has a different ``partitioning`` flag.

        Examples:
            ```py title="Appending two GasSpecies objects"
            species1.append(species2)
            # species1 now represents both original species
            ```
        """
        if not isinstance(other, GasSpecies):
            raise TypeError("Argument 'other' must be a GasSpecies object.")

        # helper: promote scalar -> 1-D numpy array
        def _as_array(val, dtype):
            return (
                val
                if isinstance(val, np.ndarray)
                else np.array([val], dtype=dtype)
            )

        # concatenate/extend every attribute
        self.name = np.concatenate(
            [_as_array(self.name, np.str_), _as_array(other.name, np.str_)]
        )
        self.molar_mass = np.concatenate(
            [
                _as_array(self.molar_mass, np.float64),
                _as_array(other.molar_mass, np.float64),
            ]
        )
        self.concentration = np.concatenate(
            [
                _as_array(self.concentration, np.float64),
                _as_array(other.concentration, np.float64),
            ]
        )
        if self.partitioning != other.partitioning:
            raise ValueError(
                "Cannot append GasSpecies with different 'partitioning' flags"
            )

        # always keep strategies in a list, then extend
        if not isinstance(self.pure_vapor_pressure_strategy, list):
            self.pure_vapor_pressure_strategy = [
                self.pure_vapor_pressure_strategy
            ]
        if not isinstance(other.pure_vapor_pressure_strategy, list):
            other_strategies = [other.pure_vapor_pressure_strategy]
        else:
            other_strategies = other.pure_vapor_pressure_strategy
        self.pure_vapor_pressure_strategy.extend(other_strategies)

    def _check_if_negative_concentration(
        self, values: Union[float, NDArray[np.float64]]
    ) -> Union[float, NDArray[np.float64]]:
        """Ensure concentration is not negative.

        Arguments:
            - values : Concentration values to check.

        Returns:
            - Corrected concentration (≥ 0).
        """
        if np.any(values < 0.0):
            message = "Negative concentration in gas species, set = 0."
            logger.warning(message)
            warnings.warn(message, UserWarning, stacklevel=2)
            # Set negative concentrations to 0
            values = np.maximum(values, 0.0)
        return values

    def _check_non_positive_value(
        self, value: Union[float, NDArray[np.float64]], name: str
    ) -> None:
        """Raise an error if any value is non-positive.

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
