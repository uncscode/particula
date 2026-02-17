"""Gas Species module.

Units are in kg/mol for molar mass, Kelvin for temperature, Pascals for
pressure, and kg/m^3 for concentration.
"""

import copy
import logging
from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray

from particula.gas.gas_data import GasData
from particula.gas.vapor_pressure_strategies import (
    ConstantVaporPressureStrategy,
    VaporPressureStrategy,
)
from particula.util.validate_inputs import validate_inputs

logger = logging.getLogger("particula")

_DEPRECATION_MESSAGE = (
    "GasSpecies is deprecated. Use GasData instead. "
    "See migration guide: docs/Features/particle-data-migration.md"
)


def _warn_deprecated(*, stacklevel: int = 2) -> None:
    """Log a deprecation notice at INFO level.

    Uses ``logger.info`` instead of ``warnings.warn`` so the message is
    visible during normal operation but never triggers failures under
    ``-Werror`` or ``pytest -W error``.

    Args:
        stacklevel: Unused, kept for call-site compatibility.
    """
    logger.info(_DEPRECATION_MESSAGE)


class GasSpecies:
    """Facade for gas species behavior backed by GasData.

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

    _data: GasData
    _single_species_name_mode: str
    _single_species_molar_mass_mode: str
    _single_species_concentration_mode: Optional[str]

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
        """Initialize with name, molar mass, and vapor pressure strategy.

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
        _warn_deprecated(stacklevel=2)

        self.pure_vapor_pressure_strategy = vapor_pressure_strategy

        n_species = self._infer_species_count(name, molar_mass, concentration)
        names, name_mode = self._normalize_names(name, n_species)
        molar_mass_array, molar_mass_mode = self._normalize_molar_mass(
            molar_mass, n_species
        )
        concentration_array, concentration_mode = self._normalize_concentration(
            concentration, n_species
        )
        concentration_array = np.asarray(
            self._check_if_negative_concentration(concentration_array),
            dtype=np.float64,
        )
        partitioning_array = self._normalize_partitioning(
            partitioning, n_species
        )

        self._data = GasData(
            name=names,
            molar_mass=molar_mass_array,
            concentration=concentration_array,
            partitioning=partitioning_array,
        )
        self._single_species_name_mode = name_mode
        self._single_species_molar_mass_mode = molar_mass_mode
        self._single_species_concentration_mode = concentration_mode

    @classmethod
    def from_data(
        cls,
        data: GasData,
        vapor_pressure_strategy: Union[
            VaporPressureStrategy, list[VaporPressureStrategy]
        ],
    ) -> "GasSpecies":
        """Create a facade from GasData without a deprecation warning.

        Args:
            data: GasData instance to wrap.
            vapor_pressure_strategy: Vapor pressure strategy or list of
                strategies to retain behavior on the facade.

        Returns:
            GasSpecies facade over the provided GasData.
        """
        unique_partitioning = np.unique(data.partitioning)
        if len(unique_partitioning) > 1:
            part_list = data.partitioning.tolist()
            raise ValueError(
                "GasData has mixed partitioning values "
                f"{part_list}, but GasSpecies requires uniform partitioning"
            )
        instance = cls.__new__(cls)
        instance.pure_vapor_pressure_strategy = vapor_pressure_strategy
        instance._data = data
        if data.n_species == 1:
            instance._single_species_name_mode = "str"
            instance._single_species_molar_mass_mode = "scalar"
            if data.n_boxes == 1:
                instance._single_species_concentration_mode = "scalar"
            else:
                instance._single_species_concentration_mode = "array"
        else:
            instance._single_species_name_mode = "array"
            instance._single_species_molar_mass_mode = "array"
            instance._single_species_concentration_mode = None
        return instance

    @property
    def data(self) -> GasData:
        """Return the underlying GasData instance."""
        return self._data

    @property
    def name(self) -> Union[str, NDArray[np.str_]]:
        """Gas species name(s)."""  # noqa: D402
        if (
            self._data.n_species == 1
            and self._single_species_name_mode == "str"
        ):
            return self._data.name[0]
        return np.asarray(self._data.name, dtype=np.str_)

    @property
    def molar_mass(self) -> Union[float, NDArray[np.float64]]:
        """Return the molar mass in kg/mol."""
        if (
            self._data.n_species == 1
            and self._single_species_molar_mass_mode == "scalar"
        ):
            return float(self._data.molar_mass[0])
        return np.asarray(self._data.molar_mass, dtype=np.float64)

    @property
    def partitioning(self) -> bool:
        """Return the partitioning flag (True if the species can partition)."""
        return bool(self._data.partitioning[0])

    @property
    def concentration(self) -> Union[float, NDArray[np.float64]]:
        """Return the concentration of the gas species in kg/m^3."""
        if self._data.n_species == 1:
            if self._single_species_concentration_mode == "scalar":
                return float(self._data.concentration[0, 0])
            return np.asarray(self._data.concentration[:, 0], dtype=np.float64)
        return np.asarray(self._data.concentration[0, :], dtype=np.float64)

    @concentration.setter
    def concentration(
        self, new_concentration: Union[float, NDArray[np.float64]]
    ):
        self._set_concentration(new_concentration)

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
        return self._data.n_species

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
            concentration, molar_mass = self._species_arrays_for_strategy()
            return np.array(
                [
                    strategy.partial_pressure(
                        concentration=c, molar_mass=m, temperature=temperature
                    )
                    for (strategy, c, m) in zip(
                        self.pure_vapor_pressure_strategy,
                        concentration,
                        molar_mass,
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
            concentration, molar_mass = self._species_arrays_for_strategy()
            return np.array(
                [
                    strategy.saturation_ratio(
                        concentration=c, molar_mass=m, temperature=temperature
                    )
                    for (strategy, c, m) in zip(
                        self.pure_vapor_pressure_strategy,
                        concentration,
                        molar_mass,
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
            _, molar_mass = self._species_arrays_for_strategy()
            return np.array(
                [
                    strategy.saturation_concentration(
                        molar_mass=m, temperature=temperature
                    )
                    for (strategy, m) in zip(
                        self.pure_vapor_pressure_strategy,
                        molar_mass,
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
        self._set_concentration(new_concentration)

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

        if self.partitioning != other.partitioning:
            raise ValueError(
                "Cannot append GasSpecies with different 'partitioning' flags"
            )

        self_names = list(self._data.name)
        other_names = list(other._data.name)
        new_names = self_names + other_names

        new_molar_mass = np.concatenate(
            [self._data.molar_mass, other._data.molar_mass]
        )

        if self._data.n_boxes == other._data.n_boxes:
            left_concentration = self._data.concentration
            right_concentration = other._data.concentration
        elif self._data.n_boxes == 1:
            left_concentration = np.tile(
                self._data.concentration, (other._data.n_boxes, 1)
            )
            right_concentration = other._data.concentration
        elif other._data.n_boxes == 1:
            left_concentration = self._data.concentration
            right_concentration = np.tile(
                other._data.concentration, (self._data.n_boxes, 1)
            )
        else:
            raise ValueError(
                "Cannot append GasSpecies with different box dimensions"
            )

        new_concentration = np.concatenate(
            [left_concentration, right_concentration], axis=1
        )
        new_partitioning = np.concatenate(
            [self._data.partitioning, other._data.partitioning]
        )

        new_data = GasData(
            name=new_names,
            molar_mass=new_molar_mass,
            concentration=new_concentration,
            partitioning=new_partitioning,
        )

        self._data = new_data
        self._single_species_name_mode = "array"
        self._single_species_molar_mass_mode = "array"
        self._single_species_concentration_mode = None

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

    def _infer_species_count(
        self,
        name: Union[str, NDArray[np.str_]],
        molar_mass: Union[float, NDArray[np.float64]],
        concentration: Union[float, NDArray[np.float64]],
    ) -> int:
        """Infer the number of species from the provided inputs."""
        name_count = 1 if isinstance(name, str) else len(name)
        species_counts = [name_count]
        for values in (molar_mass, concentration):
            array = np.asarray(values)
            if array.ndim == 1:
                species_counts.append(array.size)
            elif array.ndim == 2:
                species_counts.append(array.shape[1])
        return max(species_counts)

    def _normalize_names(
        self, name: Union[str, NDArray[np.str_]], n_species: int
    ) -> tuple[list[str], str]:
        """Normalize name input and capture return mode."""
        if isinstance(name, str):
            if n_species == 1:
                return [name], "str"
            return [name] * n_species, "array"
        if isinstance(name, np.ndarray):
            return [str(item) for item in name.tolist()], "array"
        return list(name), "array"

    def _normalize_molar_mass(
        self, molar_mass: Union[float, NDArray[np.float64]], n_species: int
    ) -> tuple[NDArray[np.float64], str]:
        """Normalize molar mass input and capture return mode."""
        molar_mass_array = np.asarray(molar_mass, dtype=np.float64)
        if molar_mass_array.ndim == 0:
            if n_species > 1:
                molar_mass_array = np.full(
                    n_species, molar_mass_array.item(), dtype=np.float64
                )
            else:
                molar_mass_array = np.array(
                    [molar_mass_array.item()], dtype=np.float64
                )
            return molar_mass_array, "scalar"
        molar_mass_array = molar_mass_array.reshape(-1)
        if molar_mass_array.size == 1 and n_species > 1:
            molar_mass_array = np.full(
                n_species, molar_mass_array.item(), dtype=np.float64
            )
        if molar_mass_array.size != n_species:
            raise ValueError(
                "molar_mass length does not match number of species: "
                f"got {molar_mass_array.size}, expected {n_species}"
            )
        return molar_mass_array, "array"

    def _normalize_concentration(
        self,
        concentration: Union[float, NDArray[np.float64]],
        n_species: int,
    ) -> tuple[NDArray[np.float64], Optional[str]]:
        """Normalize concentration input to GasData shape."""
        conc_array = np.asarray(concentration, dtype=np.float64)
        if conc_array.ndim == 0:
            return self._normalize_concentration_scalar(conc_array, n_species)
        if conc_array.ndim == 1:
            return self._normalize_concentration_vector(conc_array, n_species)
        if conc_array.ndim == 2:
            return self._normalize_concentration_matrix(conc_array, n_species)
        raise ValueError("concentration must be scalar, 1D, or 2D")

    def _normalize_concentration_scalar(
        self, conc_array: NDArray[np.float64], n_species: int
    ) -> tuple[NDArray[np.float64], Optional[str]]:
        """Normalize scalar concentration inputs."""
        value = conc_array.item()
        if n_species == 1:
            scalar = np.array([value], dtype=np.float64)
            return scalar.reshape(1, 1), "scalar"
        filled = np.full(n_species, value, dtype=np.float64)
        return filled.reshape(1, -1), None

    def _normalize_concentration_vector(
        self, conc_array: NDArray[np.float64], n_species: int
    ) -> tuple[NDArray[np.float64], Optional[str]]:
        """Normalize 1D concentration inputs."""
        conc_array = conc_array.reshape(-1)
        if n_species == 1:
            if conc_array.size == 1:
                return conc_array.reshape(1, 1), "array"
            return conc_array.reshape(-1, 1), "array"
        if conc_array.size == 1:
            conc_array = np.full(n_species, conc_array.item(), dtype=np.float64)
        if conc_array.size != n_species:
            raise ValueError(
                "concentration length does not match number of species: "
                f"got {conc_array.size}, expected {n_species}"
            )
        return conc_array.reshape(1, -1), None

    def _normalize_concentration_matrix(
        self, conc_array: NDArray[np.float64], n_species: int
    ) -> tuple[NDArray[np.float64], Optional[str]]:
        """Normalize 2D concentration inputs."""
        if conc_array.shape[1] != n_species:
            raise ValueError(
                "concentration shape does not match number of species: "
                f"got {conc_array.shape[1]}, expected {n_species}"
            )
        if n_species > 1 and conc_array.shape[0] != 1:
            raise ValueError(
                "GasSpecies only supports a single box for multi-species"
            )
        return conc_array, "array" if n_species == 1 else None

    def _normalize_partitioning(
        self, partitioning: bool, n_species: int
    ) -> NDArray[np.bool_]:
        """Normalize partitioning input to boolean array."""
        return np.array([partitioning] * n_species, dtype=np.bool_)

    def _set_concentration(
        self, new_concentration: Union[float, NDArray[np.float64]]
    ) -> None:
        """Set concentration by rebuilding GasData atomically."""
        concentration_array, concentration_mode = self._normalize_concentration(
            new_concentration, self._data.n_species
        )
        concentration_array = np.asarray(
            self._check_if_negative_concentration(concentration_array),
            dtype=np.float64,
        )
        new_data = GasData(
            name=list(self._data.name),
            molar_mass=np.copy(self._data.molar_mass),
            concentration=concentration_array,
            partitioning=np.copy(self._data.partitioning),
        )
        self._data = new_data
        self._single_species_concentration_mode = concentration_mode

    def _species_arrays_for_strategy(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return concentration and molar mass arrays for strategy.

        The arrays are shaped for vapor-pressure strategy evaluation.
        """
        if self._data.n_species == 1:
            concentration = np.array(
                [self._data.concentration[0, 0]], dtype=np.float64
            )
            molar_mass = np.array([self._data.molar_mass[0]], dtype=np.float64)
            return concentration, molar_mass
        return (
            np.asarray(self._data.concentration[0, :], dtype=np.float64),
            np.asarray(self._data.molar_mass, dtype=np.float64),
        )

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
