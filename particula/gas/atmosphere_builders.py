"""A builder class for creating Atmosphere objects with validation,
unit conversion, and a fluent interface.
"""

import copy
import logging

from particula.abc_builder import (
    BuilderABC,
)
from particula.builder_mixin import (
    BuilderPressureMixin,
    BuilderTemperatureMixin,
)
from particula.gas.atmosphere import Atmosphere
from particula.gas.species import GasSpecies

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
        - temperature : Temperature of the gas mixture in Kelvin.
        - total_pressure : Total pressure of the gas mixture in Pascals.
        - partitioning_species : GasSpecies object(s) representing partitioning
          species (or None if not set).
        - gas_only_species : GasSpecies object(s) representing non-partitioning
          species (or None if not set).

    Methods:
    - set_temperature : Set the temperature (with optional unit handling).
    - set_pressure : Set the total pressure (with optional unit handling).
    - add_partitioning_species : Add a partitioning GasSpecies object to the
      gas mixture.
    - add_gas_only_species : Add a non-partitioning GasSpecies object to the
      gas mixture.
    - set_parameters : Set multiple parameters from a dictionary.
    - build : Validate parameters and return an Atmosphere object.

    Example:
        ```py title="Create an atmosphere using the builder"
        import particula as par
        builder = par.gas.AtmosphereBuilder()
        o2 = par.gas.GasSpecies(
            name="O2", molar_mass=0.032, partitioning=True
        )
        n2 = par.gas.GasSpecies(
            name="N2", molar_mass=0.028, partitioning=False
        )
        atmosphere = (
            builder.set_temperature(300, "K")
            .set_pressure(101325, "Pa")
            .set_more_partitioning_species(o2)
            .set_more_gas_only_species(n2)
            .build()
        )
        ```
    """

    def __init__(self):
        """Initialize the Atmosphere builder.

        Sets up the builder with required parameters for creating an
        Atmosphere object, including temperature and pressure.
        """
        required_parameters = [
            "temperature",
            "pressure",
        ]
        BuilderABC.__init__(self, required_parameters)
        BuilderPressureMixin.__init__(self)
        BuilderTemperatureMixin.__init__(self)
        self.partitioning_species: GasSpecies | None = None
        self.gas_only_species: GasSpecies | None = None

    def set_more_partitioning_species(
        self, species: GasSpecies
    ) -> "AtmosphereBuilder":
        """Add a partitioning GasSpecies object to the mixture.

        Arguments:
            - species : The GasSpecies instance to add (must have
              ``partitioning=True``).

        Returns:
            - AtmosphereBuilder : This builder (for chaining).

        Raises:
            - ValueError : If ``species.partitioning`` is False.
        """
        if not species.get_partitioning():
            raise ValueError("Provided species has partitioning=False")
        if self.partitioning_species is None:
            # store a copy to avoid external mutation
            self.partitioning_species = copy.deepcopy(species)
        else:
            self.partitioning_species.append(species)
        return self

    def set_more_gas_only_species(
        self, species: GasSpecies
    ) -> "AtmosphereBuilder":
        """Add a non-partitioning GasSpecies object to the mixture.

        Arguments:
            - species : The GasSpecies instance to add (must have
              ``partitioning=False``).

        Returns:
            - AtmosphereBuilder : This builder (for chaining).

        Raises:
            - ValueError : If ``species.partitioning`` is False.
        """
        if species.get_partitioning():
            raise ValueError("Provided species has partitioning=True")
        if self.gas_only_species is None:
            self.gas_only_species = copy.deepcopy(species)
        else:
            self.gas_only_species.append(species)
        return self

    def build(self) -> Atmosphere:
        """Validate the configuration and construct the Atmosphere object.

        This method checks that all necessary conditions are met for a valid
        Atmosphere instance (e.g., at least one partitioning and one gas-only
        species must be present) and then initializes the Atmosphere.

        Returns:
            - Atmosphere : The newly created Atmosphere object.

        Raises:
            - ValueError : If no partitioning or gas-only species have been
              added to the mixture.
        """
        self.pre_build_check()
        return Atmosphere(
            temperature=self.temperature,  # type: ignore
            total_pressure=self.pressure,  # type: ignore
            partitioning_species=self.partitioning_species,  # type: ignore
            gas_only_species=self.gas_only_species,  # type: ignore
        )
