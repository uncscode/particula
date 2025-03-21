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
    """
    Builder class for creating Atmosphere objects using a fluent interface.

    This class provides methods to configure and build an Atmosphere object,
    allowing for step-by-step setting of atmospheric properties and
    species composition.

    Attributes:
        - temperature : Temperature of the gas mixture in Kelvin.
        - total_pressure : Total pressure of the gas mixture in Pascals.
        - species : List of GasSpecies objects in the mixture (starts empty).

    Methods:
    - set_temperature : Set the temperature (with optional unit handling).
    - set_pressure : Set the total pressure (with optional unit handling).
    - add_species : Add a GasSpecies object to the gas mixture.
    - set_parameters : Set multiple parameters from a dictionary.
    - build : Validate parameters and return an Atmosphere object.

    Example:
        ```py title="Create an atmosphere using the builder"
        import particula as par
        builder = par.gas.AtmosphereBuilder()
        atmosphere = (
            builder.set_temperature(300, "K")
            .set_pressure(101325, "Pa")
            .add_species(par.gas.GasSpecies(name="O2", molar_mass=0.032))
            .build()
        )
        ```
    """

    def __init__(self):
        required_parameters = ["temperature", "pressure", "species"]
        BuilderABC.__init__(self, required_parameters)
        BuilderPressureMixin.__init__(self)
        BuilderTemperatureMixin.__init__(self)
        self.species: list[GasSpecies] = []

    def add_species(self, species: GasSpecies) -> "AtmosphereBuilder":
        """
        Add a GasSpecies object to the gas mixture.

        Arguments:
            - species : The GasSpecies object to be added.

        Returns:
            - AtmosphereBuilder : This builder, for method chaining.
        """
        self.species.append(species)
        return self

    def build(self) -> Atmosphere:
        """
        Validate the configuration and construct the Atmosphere object.

        This method checks that all necessary conditions are met for a valid
        Atmosphere instance (e.g., at least one species must be present) and
        then initializes the Atmosphere.

        Returns:
            - Atmosphere : The newly created Atmosphere object.

        Raises:
            - ValueError : If no species have been added to the mixture.
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
