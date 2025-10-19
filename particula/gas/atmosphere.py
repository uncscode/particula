"""Atmosphere module for modeling gas mixtures in a given environment."""

from dataclasses import dataclass

from particula.gas.species import GasSpecies


@dataclass
class Atmosphere:
    """Represents a mixture of gas species under specific conditions.

    This class represents the atmospheric environment by detailing properties
    such as temperature and pressure, alongside a dynamic list of gas species
    present.

    Attributes:
        - temperature : Temperature of the gas mixture in Kelvin.
        - total_pressure : Total atmospheric pressure of the mixture in
          Pascals.
        - partitioning_species : List of GasSpecies objects representing the
          various species within the gas mixture, that can be
          partitioned to the particle phase.
        - gas_only_species : List of GasSpecies objects representing the
          various species within the gas mixture, that cannot be
          partitioned to the particle phase.

    Methods:
    - add_partitioning_species : Adds a GasSpecies object to the mixture.
    - add_gas_only_species : Adds a GasSpecies object to the mixture.
    """

    temperature: float
    total_pressure: float
    partitioning_species: GasSpecies
    gas_only_species: GasSpecies

    def add_partitioning_species(self, gas_species: GasSpecies) -> None:
        """Add a GasSpecies object to the partitioning species list.

        Arguments:
            - gas_species : The gas species to be added.
        """
        self.partitioning_species.append(gas_species)

    def add_gas_only_species(self, gas_species: GasSpecies) -> None:
        """Add a GasSpecies to gas only (nonpartitioning) species list.

        Arguments:
            - gas_species : The gas species to be added.
        """
        self.gas_only_species.append(gas_species)

    def __len__(self) -> int:
        """Return the number of species in the gas mixture.

        Returns:
            - The number of gas species in the mixture.
        """
        return len(self.partitioning_species) + len(self.gas_only_species)

    def __str__(self) -> str:
        """Provide a string representation of the Atmosphere object.

        Returns:
            - Includes the temperature, pressure, and lists of partitioning and
              gas only species.
        """
        return (
            f"Gas mixture at {self.temperature} K, {self.total_pressure} Pa, "
            f"partitioning={self.partitioning_species}, "
            f"gas_only_species={self.gas_only_species}"
        )
