"""Gas module."""

from typing import List
from particula.next.gas_species import GasSpecies


class Gas:
    """
    Represents a mixture of gas species, including properties such as
    temperature, total pressure, and a list of gas species species.

    Attributes:
    - temperature (float): The temperature of the gas mixture.
    - total_pressure (float): The total pressure of the gas mixture.
    - species (List[GasSpecies]): A list of GasSpecies objects
        representing the species of the gas mixture.

    Methods:
    - add_species: Adds a gas species to the mixture.
    - remove_species: Removes a gas species from the mixture by name.

    """

    def __init__(
        self,
        temperature: float,
        total_pressure: float,
        species: list[GasSpecies],
    ):
        self.temperature = temperature
        self.total_pressure = total_pressure
        self.species = species

    def add_species(
            self,
            gas_species: GasSpecies) -> None:
        """
        Adds a gas species to the mixture.

        Parameters:
        - name (str): The name of the gas species.
        - molar_mass (float): The molar_mass of the gas species.
        - vapor_pressure (Optional[float]): The vapor pressure of the gas
            species. Units N/m2 (base, 1 kg⋅m⋅s-2) None if not applicable.
        - condensable (bool): Indicates whether the gas species is
            condensable.
        """
        self.species.append(gas_species)

    def remove_species(self, index: int) -> None:
        """
        Removes a gas species from the mixture by name.

        Parameters:
        - index int: The name of the gas species to be removed.
        """
        self.species.pop(index)


class GasBuilder:
    """A builder class for creating Gas objects with a fluent interface."""

    def __init__(self):
        self._temperature: float = 298.15
        self._total_pressure: float = 101325
        self._species: List[GasSpecies] = []

    def temperature(self, temperature: float):
        """Set the temperature of the gas mixture, in Kelvin."""
        self._temperature = temperature
        return self

    def total_pressure(self, total_pressure: float):
        """Set the total pressure of the gas mixture, in Pascals."""
        self._total_pressure = total_pressure
        return self

    def add_species(self, species: GasSpecies):
        """Add a gas species component to the gas mixture."""
        self._species.append(species)
        return self

    def build(self) -> Gas:
        """Build and return the Gas object."""
        if not self._species:
            raise ValueError("At least one gas component must be added.")
        gas = Gas(
            temperature=self._temperature,
            total_pressure=self._total_pressure,
            species=self._species,
        )
        return gas
