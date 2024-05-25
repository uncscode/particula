"""A builder class for creating Atmosphere objects with validation,
unit conversion, and a fluent interface."""

import logging
from particula.next.abc_builder import BuilderABC
from particula.next.gas.species import GasSpecies
from particula.next.gas.atmosphere import Atmosphere
from particula.util.input_handling import convert_units

logger = logging.getLogger("particula")


class AtmosphereBuilder(BuilderABC):
    """A builder class for creating Atmosphere objects with a fluent interface.

    Attributes:
    ----------
    - temperature (float): The temperature of the gas mixture in Kelvin.
    - total_pressure (float): The total pressure of the gas mixture in Pascals.
    - species (list[GasSpecies]): The list of gas species in the mixture.

    Methods:
    -------
    - set_temperature(temperature): Set the temperature of the gas mixture.
    - set_total_pressure(total_pressure): Set the total pressure of the gas
    mixture.
    - add_species(species): Add a GasSpecies component to the gas mixture.
    - set_parameters(parameters): Set the parameters from a dictionary.
    - build(): Validate and return the Atmosphere object.
    """

    def __init__(self):
        required_parameters = ['temperature', 'total_pressure', 'species']
        super().__init__(required_parameters)
        self.temperature: float = 298.15
        self.total_pressure: float = 101325
        self.species: list[GasSpecies] = []

    def set_temperature(
        self,
        temperature: float,
        temperature_units: str = "K"
    ):
        """Set the temperature of the gas mixture, in Kelvin."""
        if temperature < 0:
            logger.error("Temperature must be a positive value.")
            raise ValueError("Temperature must be a positive value.")
        self.temperature = temperature * convert_units(temperature_units, "K")
        return self

    def set_total_pressure(
        self,
        total_pressure: float,
        pressure_units: str = "Pa"
    ):
        """Set the total pressure of the gas mixture, in Pascals."""
        if total_pressure < 0:
            logger.error("Total pressure must be a positive value.")
            raise ValueError("Total pressure must be a positive value.")
        self.total_pressure = total_pressure \
            * convert_units(pressure_units, "Pa")
        return self

    def add_species(self, species: GasSpecies):
        """Add a GasSpecies component to the gas mixture."""
        self.species.append(species)
        return self

    def build(self) -> Atmosphere:
        """Validate and return the Atmosphere object."""
        if not self.species:
            raise ValueError("Atmosphere must contain at least one species.")
        self.pre_build_check()
        return Atmosphere(
            temperature=self.temperature,
            total_pressure=self.total_pressure,
            species=self.species,
        )
