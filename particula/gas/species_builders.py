"""This module contains the GasSpeciesBuilder class, which is a builder class
for GasSpecies objects. The GasSpeciesBuilder class allows for a more fluent
and readable creation of GasSpecies as this class provides validation and
unit conversion for the parameters of the GasSpecies object.
"""

from typing import Union
import logging
from numpy.typing import NDArray
import numpy as np
from particula.abc_builder import (
    BuilderABC,
)
from particula.builder_mixin import (
    BuilderMolarMassMixin,
    BuilderConcentrationMixin,
)
from particula.gas.species import GasSpecies
from particula.gas.vapor_pressure_strategies import (
    VaporPressureStrategy,
    ConstantVaporPressureStrategy,
)

logger = logging.getLogger("particula")


class GasSpeciesBuilder(
    BuilderABC, BuilderMolarMassMixin, BuilderConcentrationMixin
):
    """Builder class for GasSpecies objects, allowing for a more fluent and
    readable creation of GasSpecies instances with optional parameters.

    Attributes:
        name: The name of the gas species.
        molar_mass: The molar mass of the gas species in kg/mol.
        vapor_pressure_strategy: The vapor pressure strategy for the
            gas species.
        condensable: Whether the gas species is condensable.
        concentration: The concentration of the gas species in the
            mixture, in kg/m^3.

    Methods:
        set_name: Set the name of the gas species.
        set_molar_mass: Set the molar mass of the gas species in kg/mol.
        set_vapor_pressure_strategy: Set the vapor pressure strategy
            for the gas species.
        set_condensable: Set the condensable bool of the gas species.
        set_concentration: Set the concentration of the gas species in the
            mixture, in kg/m^3.
        set_parameters: Set the parameters of the GasSpecies object from
            a dictionary including optional units.

    Raises:
        ValueError: If any required key is missing. During check_keys and
            pre_build_check. Or if trying to set an invalid parameter.
        Warning: If using default units for any parameter.
    """

    def __init__(self):
        required_parameters = [
            "name",
            "molar_mass",
            "vapor_pressure_strategy",
            "condensable",
            "concentration",
        ]
        BuilderABC.__init__(self, required_parameters)
        BuilderMolarMassMixin.__init__(self)
        BuilderConcentrationMixin.__init__(self, default_units="kg/m^3")
        self.name = None
        self.vapor_pressure_strategy = None
        self.condensable = None

    def set_name(self, name: Union[str, NDArray[np.str_]]):
        """Set the name of the gas species."""
        self.name = name
        return self

    def set_vapor_pressure_strategy(
        self,
        strategy: Union[VaporPressureStrategy, list[VaporPressureStrategy]],
    ):
        """Set the vapor pressure strategy for the gas species."""
        self.vapor_pressure_strategy = strategy
        return self

    def set_condensable(
        self,
        condensable: Union[bool, NDArray[np.bool_]],
    ):
        """Set the condensable bool of the gas species."""
        self.condensable = condensable
        return self

    def build(self) -> GasSpecies:
        """Validate and return the GasSpecies object."""
        self.pre_build_check()
        return GasSpecies(
            name=self.name,  # type: ignore
            molar_mass=self.molar_mass,  # type: ignore
            vapor_pressure_strategy=(
                self.vapor_pressure_strategy
            ),  # type: ignore
            condensable=self.condensable,  # type: ignore
            concentration=self.concentration,  # type: ignore
        )


class PresetGasSpeciesBuilder(
    GasSpeciesBuilder,
):
    """Builder class for GasSpecies objects, allowing for a more fluent and
    readable creation of GasSpecies instances with optional parameters.

    """

    def __init__(self):
        GasSpeciesBuilder.__init__(self)
        self.name = "Preset100"
        self.molar_mass = 0.100  # kg/mol
        self.vapor_pressure_strategy = ConstantVaporPressureStrategy(
            vapor_pressure=1.0  # Pa
        )
        self.condensable = False
        self.concentration = 1.0

    def build(self) -> GasSpecies:

        self.pre_build_check()
        return GasSpecies(
            name=self.name,  # type: ignore
            molar_mass=self.molar_mass,  # type: ignore
            vapor_pressure_strategy=(
                self.vapor_pressure_strategy
            ),  # type: ignore
            condensable=self.condensable,  # type: ignore
            concentration=self.concentration,  # type: ignore
        )
