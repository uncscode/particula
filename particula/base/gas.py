"""
This module demonstrates the Composite Pattern through the modeling of gases
and their constituent species. In this pattern, both composite objects (gases)
and leaf objects (gas species) are treated uniformly.

The Composite Pattern is particularly useful in scenarios where a part-whole
hierarchy is present. In our case, a `Gas` represents the whole, which can be
composed of various parts - the `GasSpecies`. Each `GasSpecies` can have
distinct properties such as mass and vapor pressure, and the `Gas` itself can
consist of multiple species, mimicking real-world gas mixtures.

Classes:
- GasSpecies: Represents an individual gas species with properties like name,
mass, and vapor pressure. Acts as a leaf in the composite structure.
- Gas: Represents a composite object that can contain multiple GasSpecies.
It manages the collection of these species and performs operations on them as
a group.

The Composite Pattern allows for the Gas class to manage an arbitrary number
of GasSpecies objects, facilitating operations that need to consider the gas
mixture as a whole or interact with its individual components seamlessly.
"""


from typing import Optional, List
from dataclasses import dataclass, field
from numpy.typing import NDArray
import numpy as np


@dataclass
class GasSpecies:
    """
    Represents a single species of gas, including its properties such as
    name, mass, vapor pressure, and whether it is condensable.

    Attributes:
    - name (str): The name of the gas species.
    - mass (float): The mass of the gas species.
    - vapor_pressure (Optional[float]): The vapor pressure of the gas
        species. None if not applicable.
    - condensable (bool): Indicates whether the gas species is condensable.
    """
    name: str
    mass: float
    vapor_pressure: Optional[float] = None
    condensable: bool = field(default=False)

    def get_mass(self) -> np.float64:
        """
        Returns the mass of the gas species as an np.float64.

        Returns:
            np.float64: The mass of the gas species.
        """
        return np.float64(self.mass)

    def is_condensable(self) -> bool:
        """
        Checks if the gas species is condensable.

        Returns:
            bool: True if the gas species is condensable, False otherwise.
        """
        return self.condensable

    def get_mass_condensable(self) -> np.float64:
        """
        Returns the mass of the gas species if it is condensable,
        otherwise returns 0.

        Returns:
            np.float64: The mass of the gas species if it is condensable,
            otherwise 0.
        """
        return np.float64(self.mass) if self.condensable else np.float64(0)


@dataclass
class Gas:
    """
    Represents a mixture of gas species, including properties such as
    temperature, total pressure, and a list of gas species components.

    Attributes:
    - temperature (float): The temperature of the gas mixture.
    - total_pressure (float): The total pressure of the gas mixture.
    - components (List[GasSpecies]): A list of GasSpecies objects
        representing the components of the gas mixture.

    Methods:
    - add_species: Adds a gas species to the mixture.
    - remove_species: Removes a gas species from the mixture by name.
    - get_mass: Returns the mass of a specified species or the masses of all
        species in the gas mixture as an np.ndarray.
    - get_mass_condensable: Returns the mass of a specific condensable species
        or the masses of all condensable species in the gas mixture as an
        np.ndarray.

    """
    temperature: float = 298.15
    total_pressure: float = 101325
    components: List[GasSpecies] = field(default_factory=list)

    def add_species(
            self,
            name: str,
            mass: float,
            vapor_pressure: Optional[float] = None,
            condensable: bool = False) -> None:
        """
        Adds a gas species to the mixture.

        Parameters:
        - name (str): The name of the gas species.
        - mass (float): The mass of the gas species.
        - vapor_pressure (Optional[float]): The vapor pressure of the gas
            species. None if not applicable.
        - condensable (bool): Indicates whether the gas species is
            condensable.
        """
        species = GasSpecies(name, mass, vapor_pressure, condensable)
        self.components.append(species)

    def remove_species(self, name: str) -> None:
        """
        Removes a gas species from the mixture by name.

        Parameters:
        - name (str): The name of the gas species to be removed.
        """
        self.components = [c for c in self.components if c.name != name]

    def get_mass(self, name: Optional[str] = None) -> NDArray[np.float64]:
        """
        Returns the mass of a specified species or the masses of all species
        in the gas mixture as an np.ndarray.

        If a name is specified, the method returns an array containing the
        mass of the named species. If the specified name is not found in the
        mixture, a ValueError is raised. If no name is specified, it returns
        an array of the masses of all species in the mixture.

        Parameters:
            name (Optional[str]): The name of the gas species for which to
            return the mass. If None, returns masses of all species.

        Returns:
            NDArray[np.float64]: An array containing the requested mass(es).

        Raises:
            ValueError: If the specified name is not found in the mixture.
        """
        if name:
            matching_masses = [component.get_mass()
                               for component in self.components
                               if component.name == name]
            if not matching_masses:
                raise ValueError(
                    f"Gas species '{name}' not found in the mixture.")
            return np.array(matching_masses, dtype=np.float64)

        # Return the masses of all components if no name is specified.
        masses = [component.get_mass() for component in self.components]
        return np.array(masses, dtype=np.float64)

    def get_mass_condensable(
            self, name: Optional[str] = None) -> NDArray[np.float64]:
        """
        Returns the mass of a specific condensable species or the masses of
        all condensable species in the gas mixture as an np.ndarray. If a name
        is provided, only the mass of that specific condensable species is
        returned.

        Parameters:
            name (Optional[str]): The name of the specific condensable gas
            species to retrieve the mass for. If None (the default), the
            masses of all condensable species are returned.

        Returns:
            NDArray[np.float64]: An array of the mass of the specified
            condensable species, or an array of the masses of all condensable
            species in the gas mixture.

        Raises:
            ValueError: If a specific species name is provided but not found
            in the mixture, or if it's not condensable.
        """
        if name:
            for component in self.components:
                if component.name == name:
                    if component.is_condensable():
                        return np.array(
                            [component.get_mass()], dtype=np.float64)
                    raise ValueError(
                        f"Gas species '{name}' is not condensable.")
            raise ValueError(f"Gas species '{name}' not found in the mixture.")

        masses_condensable = [component.get_mass(
        ) for component in self.components if component.is_condensable()]
        return np.array(masses_condensable, dtype=np.float64)
