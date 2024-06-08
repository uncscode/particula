"""This module contains the GasSpeciesBuilder class, which is a builder class
for GasSpecies objects. The GasSpeciesBuilder class allows for a more fluent
and readable creation of GasSpecies as this class provides validation and
unit conversion for the parameters of the GasSpecies object.
"""

from typing import Union
import logging
from numpy.typing import NDArray
import numpy as np
from particula.next.abc_builder import (
    BuilderABC,
    BuilderMolarMassMixin,
    BuilderConcentrationMixin,
)
from particula.next.gas.species import GasSpecies
from particula.next.gas.vapor_pressure_strategies import (
    VaporPressureStrategy,
)

logger = logging.getLogger("particula")


class GasSpeciesBuilder(
    BuilderABC, BuilderMolarMassMixin, BuilderConcentrationMixin
):
    """Builder class for GasSpecies objects, allowing for a more fluent and
    readable creation of GasSpecies instances with optional parameters.

    Attributes:
    ----------
    - name (str): The name of the gas species.
    - molar_mass (float): The molar mass of the gas species in kg/mol.
    - vapor_pressure_strategy (VaporPressureStrategy): The vapor pressure
        strategy for the gas species.
    - condensable (bool): Whether the gas species is condensable.
    - concentration (float): The concentration of the gas species in the
        mixture, in kg/m^3.

    Methods:
    -------
    - set_name(name): Set the name of the gas species.
    - set_molar_mass(molar_mass, molar_mass_units): Set the molar mass of the
        gas species in kg/mol.
    - set_vapor_pressure_strategy(strategy): Set the vapor pressure strategy
        for the gas species.
    - set_condensable(condensable): Set the condensable bool of the gas
        species.
    - set_concentration(concentration, concentration_units): Set the
        concentration of the gas species in the mixture, in kg/m^3.
    - set_parameters(params): Set the parameters of the GasSpecies object from
        a dictionary including optional units.
    - build(): Validate and return the GasSpecies object.

    Raises:
    ------
    - ValueError: If any required key is missing. During check_keys and
        pre_build_check. Or if trying to set an invalid parameter.
    - Warning: If using default units for any parameter.
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
            vapor_pressure_strategy=self.vapor_pressure_strategy,  # type: ignore
            condensable=self.condensable,  # type: ignore
            concentration=self.concentration,  # type: ignore
        )
