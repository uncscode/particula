"""Atmosphere module for modeling gas mixtures in a given environment."""

from dataclasses import dataclass, field
from particula.gas.species import GasSpecies


@dataclass
class Atmosphere:
    """Represents a mixture of gas species under specific conditions.

    This class represents the atmospheric environment by detailing properties
    such as temperature and pressure, alongside a dynamic list of gas species
    present.

    Attributes:
        temperature: Temperature of the gas mixture in Kelvin.
        total_pressure: Total atmospheric pressure of the mixture inPascals.
        species: List of GasSpecies objects representing the
            various species within the gas mixture.

    Methods:
        add_species(self, species: GasSpecies) -> None:
            Adds a GasSpecies object to the mixture.
        remove_species(self, index: int) -> None:
            Removes a GasSpecies object from the mixture based on its index.
    """

    temperature: float
    total_pressure: float
    species: list[GasSpecies] = field(default_factory=list)

    def add_species(self, gas_species: GasSpecies) -> None:
        """Adds a GasSpecies object to the mixture.

        Args:
            gas_species: The gas species to be added.
        """
        self.species.append(gas_species)

    def remove_species(self, index: int) -> None:
        """Removes a gas species from the mixture by its index.

        Args:
            index: Index of the gas species to remove. Must be within
                        the current range of the list.

        Raises:
            IndexError: If the provided index is out of bounds.
        """
        if 0 <= index < len(self.species):
            self.species.pop(index)
        else:
            raise IndexError("No gas species at the provided index.")

    def __iter__(self):
        """
        Allows iteration over the species in the gas mixture.

        Returns:
            Iterator[GasSpecies]: An iterator over the gas species objects.
        """
        return iter(self.species)

    def __getitem__(self, index: int) -> GasSpecies:
        """Retrieves a gas species by index.

        Args:
            index: The index of the gas species to retrieve.

        Returns:
            GasSpecies: The gas species at the specified index.
        """
        return self.species[index]

    def __len__(self) -> int:
        """Returns the number of species in the gas mixture.

        Returns:
            int: The number of gas species in the mixture.
        """
        return len(self.species)

    def __str__(self) -> str:
        """Provides a string representation of the Atmosphere object.

        Returns:
            str: A string that includes the temperature, pressure, and a
                list of species in the mixture.
        """
        return (
            f"Gas mixture at {self.temperature} K and {self.total_pressure} Pa"
            f" consisting of {[str(species) for species in self.species]}"
        )
