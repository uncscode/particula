"""Aerosol class just a list of gas classes and particle classes.

There is a problem here, with matching of gases that can condense to,
particles and getting it to work correctly. This is will be solved
with usage as we figure out the best way to do this.
"""
from typing import List, Union, Iterator
from particula.next.gas import Gas, GasSpecies
from particula.next.particle import Particle


class Aerosol:
    """
    A class for interacting with collections of Gas and Particle objects.
    Allows for the representation and manipulation of an aerosol, which
    is composed of various gases and particles.
    """

    def __init__(self, gas: Gas,
                 particles: Union[Particle, List[Particle]]):
        """
        Initializes an Aerosol instance with Gas and Particle instances.

        Parameters:
        - gas (Gas): Gas with many GasSpeices possible.
        - particles (Union[Particle, List[Particle]]): One or more Particle
        instances.
        """
        self.gas = gas
        self.particles: List[Particle] = [particles] if isinstance(
            particles, Particle) else particles

    def iterate_gas(self) -> Iterator[GasSpecies]:
        """
        Returns an iterator for gas species.

        Returns:
        Iterator[GasSpecies]: An iterator over the gas species type.
        """
        return iter(self.gas)

    def iterate_particle(self) -> Iterator[Particle]:
        """
        Returns an iterator for particle.

        Returns:
        Iterator[Particle]: An iterator over the particle type.
        """
        return iter(self.particles)

    def add_gas(self, gas: Gas):
        """
        Replaces the current Gas instance with a new one.

        Parameters:
        - gas (Gas): The Gas instance to replace the current one.
        """
        self.gas = gas

    def add_particle(self, particle: Particle):
        """
        Adds a Particle instance to the aerosol.

        Parameters:
        - particle (Particle): The Particle instance to add.
        """
        self.particles.append(particle)
