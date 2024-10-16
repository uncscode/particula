"""A builder class for creating Atmosphere objects with validation,
unit conversion, and a fluent interface."""

import logging
from particula.abc_builder import (
    BuilderABC,
)
from particula.builder_mixin import (
    BuilderPressureMixin,
    BuilderTemperatureMixin,
)
from particula.gas.species import GasSpecies
from particula.gas.atmosphere import Atmosphere

logger = logging.getLogger("particula")


class AtmosphereBuilder(
    BuilderABC,
    BuilderTemperatureMixin,
    BuilderPressureMixin,
):
    """Builder class for creating Atmosphere objects using a fluent interface.

    This class provides methods to configure and build an Atmosphere object,
    allowing for step-by-step setting of atmospheric properties and
    species composition.

    Attributes:
        temperature: Temperature of the gas mixture in Kelvin.
        total_pressure (float): Total pressure of the gas mixture in Pascals.
        species (list[GasSpecies]): List of GasSpecies objects in the mixture.
            Starts empty.

    Methods:
        set_temperature(temperature,temperature_units): Sets the temperature.
        set_pressure(pressure,pressure_units): Sets the total pressure.
        add_species(species): Adds a GasSpecies object to the gas mixture.
        set_parameters(parameters): Sets multiple parameters from a dictionary.
        build(): Validates the set parameters and returns an Atmosphere object.
    """

    def __init__(self):
        required_parameters = ["temperature", "pressure", "species"]
        BuilderABC.__init__(self, required_parameters)
        BuilderPressureMixin.__init__(self)
        BuilderTemperatureMixin.__init__(self)
        self.species: list[GasSpecies] = []

    def add_species(self, species: GasSpecies) -> "AtmosphereBuilder":
        """Adds a GasSpecies object to the gas mixture.

        Args:
            species (GasSpecies): The GasSpecies object to be added.

        Returns:
            AtmosphereBuilder: Instance of this builder for chaining.
        """
        self.species.append(species)
        return self

    def build(self) -> Atmosphere:
        """Validates the configuration and constructs the Atmosphere object.

        This method checks that all necessary conditions are met for a valid
        Atmosphere instance(e.g., at least one species must be present) and
        then initializes the Atmosphere.

        Returns:
            Atmosphere: The newly created Atmosphere object, configured as
            specified.

        Raises:
            ValueError: If no species have been added to the mixture.
        """
        if not self.species:  # we may remove this check
            message = "Atmosphere must contain at least one species."
            logger.error(message)
            raise ValueError(message)
        self.pre_build_check()
        return Atmosphere(
            temperature=self.temperature,  # type: ignore
            total_pressure=self.pressure,  # type: ignore
            species=self.species,
        )
