"""Atmosphere module for modeling gas mixtures in a given environment."""

from dataclasses import dataclass, field
from particula.gas.species import GasSpecies


@dataclass
class Atmosphere:
    """
    Represents a mixture of gas species under specific conditions.

    This class represents the atmospheric environment by detailing properties
    such as temperature and pressure, alongside a dynamic list of gas species
    present.

    Attributes:
        - temperature : Temperature of the gas mixture in Kelvin.
        - total_pressure : Total atmospheric pressure of the mixture in
          Pascals.
        - species : List of GasSpecies objects representing the
            various species within the gas mixture.

    Methods:
    - add_species : Adds a GasSpecies object to the mixture.
    - remove_species : Removes a GasSpecies object from the mixture by index.
    """

    temperature: float
    total_pressure: float
    species: list[GasSpecies] = field(default_factory=list)

    def add_species(self, gas_species: GasSpecies) -> None:
        """
        Add a GasSpecies object to the mixture.

        Arguments:
            - gas_species : The gas species to be added.
        """
        self.species.append(gas_species)

    def remove_species(self, index: int) -> None:
        """
        Remove a gas species from the mixture by its index.

        Arguments:
            - index : Index of the gas species to remove. Must be in range.

        Raises:
            - IndexError : If the provided index is out of bounds.
        """
        if 0 <= index < len(self.species):
            self.species.pop(index)
        else:
            raise IndexError("No gas species at the provided index.")

    def __iter__(self):
        """
        Allow iteration over the species in the gas mixture.

        Returns:
            - Iterator[GasSpecies] : An iterator over the gas species objects.
        """
        return iter(self.species)

    def __getitem__(self, index: int) -> GasSpecies:
        """
        Retrieve a gas species by index.

        Arguments:
            - index : The index of the gas species to retrieve.

        Returns:
            - The gas species at the specified index.
        """
        return self.species[index]

    def __len__(self) -> int:
        """
        Return the number of species in the gas mixture.

        Returns:
            - The number of gas species in the mixture.
        """
        return len(self.species)

    def __str__(self) -> str:
        """
        Provide a string representation of the Atmosphere object.

        Returns:
            - Includes the temperature, pressure, and a list of species.
        """
        return (
            f"Gas mixture at {self.temperature} K and {self.total_pressure} Pa"
            f" consisting of {[str(species) for species in self.species]}"
        )
