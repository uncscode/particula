"""Gas module."""

from dataclasses import dataclass, field
from particula.next.gas.species import GasSpecies


@dataclass
class Atmosphere:
    """
    Represents a mixture of gas species, detailing properties such as
    temperature, total pressure, and the list of gas species in the mixture.

    Attributes:
    - temperature (float): The temperature of the gas mixture in Kelvin.
    - total_pressure (float): The total pressure of the gas mixture in Pascals.
    - species (List[GasSpecies]): A list of GasSpecies objects representing the
        species in the gas mixture.

    Methods:
    - add_species: Adds a gas species to the mixture.
    - remove_species: Removes a gas species from the mixture by index.
    """

    temperature: float
    total_pressure: float
    species: list[GasSpecies] = field(default_factory=list)

    def add_species(self, gas_species: GasSpecies) -> None:
        """
        Adds a gas species to the mixture.

        Parameters:
        - gas_species (GasSpecies): The GasSpecies object to be added to the
        mixture.
        """
        self.species.append(gas_species)

    def remove_species(self, index: int) -> None:
        """
        Removes a gas species from the mixture by index.

        Parameters:
        - index (int): The index of the gas species to be removed from the
        list.
        """
        if 0 <= index < len(self.species):
            self.species.pop(index)
        else:
            raise IndexError("No gas species at the provided index.")

    def __iter__(self):
        """Allows iteration over the species in the gas mixture."""
        return iter(self.species)

    def __getitem__(self, index: int) -> GasSpecies:
        """Returns the gas species at the given index."""
        return self.species[index]

    def __len__(self):
        """Returns the number of species in the gas mixture."""
        return len(self.species)

    def __str__(self):
        """Returns a string representation of the Gas object."""
        return (
            f"Gas mixture at {self.temperature} K and {self.total_pressure} Pa"
            f" consisting of {[str(species) for species in self.species]}"
        )
