"""Aerosol class just a list of gas classes and particle classes."""
from typing import List, Union, Iterator
from particula.next.gas import Gas
from particula.next.particle import Particle


class Aerosol:
    """
    A class for interacting with collections of Gas and Particle objects.
    Allows for the representation and manipulation of an aerosol, which
    is composed of various gases and particles.
    """

    def __init__(self, gases: Union[Gas, List[Gas]],
                 particles: Union[Particle, List[Particle]]):
        """
        Initializes an Aerosol instance with Gas and Particle instances.

        Parameters:
        - gases (Union[Gas, List[Gas]]): One or more Gas instances.
        - particles (Union[Particle, List[Particle]]): One or more Particle
        instances.
        """
        # Ensure gases and particles are stored as lists, even if a single
        # instance is passed
        self.gases: List[Gas] = [gases] if isinstance(gases, Gas) else gases
        self.particles: List[Particle] = [particles] if isinstance(
            particles, Particle) else particles

    def iterate(self, item_type: str) -> Iterator[Union[Gas, Particle]]:
        """
        Returns an iterator for either gases or particles based on the
        item_type parameter.

        Parameters:
        - item_type (str): Specifies the type of items to iterate over
        ('gas' or 'particle').

        Returns:
        Iterator[Union[Gas, Particle]]: An iterator over the specified type.
        """
        if item_type == 'gas':
            return iter(self.gases)
        if item_type == 'particle':
            return iter(self.particles)
        raise ValueError("Invalid item_type. Use 'gas' or 'particle'.")

    def add_gas(self, gas: Gas):
        """
        Adds a Gas instance to the aerosol.

        Parameters:
        - gas (Gas): The Gas instance to add.
        """
        self.gases.append(gas)

    def add_particle(self, particle: Particle):
        """
        Adds a Particle instance to the aerosol.

        Parameters:
        - particle (Particle): The Particle instance to add.
        """
        self.particles.append(particle)