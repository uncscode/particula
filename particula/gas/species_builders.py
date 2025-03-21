"""
This module contains the GasSpeciesBuilder class, which is a builder class
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
    """
    Builder class for GasSpecies objects with preset default parameters.

    This subclass of GasSpeciesBuilder initializes certain parameters
    (e.g., name, molar_mass, vapor_pressure_strategy, etc.) to predefined
    values. Suitable for quick testing or examples.

    Methods:
        - build : Validate parameters and return a GasSpecies object.

    Attributes:
        - name : The name of the gas species.
        - molar_mass : The molar mass of the gas species in kg/mol.
        - vapor_pressure_strategy : The vapor pressure strategy for the
            gas species.
        - condensable : Whether the gas species is condensable.
        - concentration : The concentration of the gas species in the
            mixture, in kg/m^3.

    Methods:
    - set_name : Set the name of the gas species.
    - set_vapor_pressure_strategy : Set the vapor pressure strategy.
    - set_condensable : Set whether the species is condensable.
    - set_molar_mass : From BuilderMolarMassMixin.
    - set_concentration : From BuilderConcentrationMixin.
    - build : Validate parameters and return a GasSpecies object.

    Example:
        ``` py title="Create a gas species using the builder"
        import particula as par
        builder = par.gas.GasSpeciesBuilder()
        gas_object = (
            builder.set_name("Oxygen")
            .set_molar_mass(0.032, "kg/mol")
            .set_vapor_pressure_strategy(
                par.gas.ConstantVaporPressureStrategy(vapor_pressure=101325)
            )
            .set_condensable(False)
            .set_concentration(1.2, "kg/m^3")
            .build()
        )
        # gas_object is now a GasSpecies instance with the specified
        # parameters.
        ```
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

    def set_name(
        self, name: Union[str, NDArray[np.str_]]
    ) -> "GasSpeciesBuilder":
        """
        Set the name of the gas species.

        Arguments:
            - name : The name of the gas species.

        Returns:
            - This builder instance.
        """
        self.name = name
        return self

    def set_vapor_pressure_strategy(
        self,
        strategy: Union[VaporPressureStrategy, list[VaporPressureStrategy]],
    ) -> "GasSpeciesBuilder":
        """
        Set the vapor pressure strategy for the gas species.

        Arguments:
            - strategy : The vapor pressure strategy (or list of strategies).

        Returns:
            - This builder instance.
        """
        self.vapor_pressure_strategy = strategy
        return self

    def set_condensable(
        self,
        condensable: Union[bool, NDArray[np.bool_]],
    ) -> "GasSpeciesBuilder":
        """
        Set whether the gas species is condensable.

        Arguments:
            - condensable : Boolean or array indicating condensability.

        Returns:
            - This builder instance.
        """
        self.condensable = condensable
        return self

    def build(self) -> GasSpecies:
        """
        Validate parameters and return a GasSpecies object.

        Returns:
            - The constructed GasSpecies instance.

        Raises:
            - ValueError : If any required parameters are missing or invalid.
        """
        return GasSpecies(
            name=self.name,  # type: ignore
            molar_mass=self.molar_mass,  # type: ignore
            vapor_pressure_strategy=(self.vapor_pressure_strategy),
            condensable=self.condensable,  # type: ignore
            concentration=self.concentration,  # type: ignore
        )


class PresetGasSpeciesBuilder(
    GasSpeciesBuilder,
):
    """Builder class for GasSpecies objects, allowing for a more fluent and
    readable creation of GasSpecies instances with optional parameters.

    Example:
        ``` py title="Create a gas species using the preset builder"
        import particula as par
        gas_object = par.gas.PresetGasSpeciesBuilder().build()
        # gas_object is now a GasSpecies instance with the preset
        # parameters.
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
        """
        Validate parameters and return a GasSpecies object with preset
        defaults.

        Returns:
            - GasSpecies : The constructed GasSpecies instance.
        """
        return GasSpecies(
            name=self.name,  # type: ignore
            molar_mass=self.molar_mass,  # type: ignore
            vapor_pressure_strategy=(self.vapor_pressure_strategy),
            condensable=self.condensable,  # type: ignore
            concentration=self.concentration,  # type: ignore
        )
