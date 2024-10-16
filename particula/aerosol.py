"""Aerosol class just a list of gas classes and particle classes.

There is a problem here, with matching of gases that can condense to,
particles and getting it to work correctly. This is will be solved
with usage as we figure out the best way to do this.
"""

from typing import List, Union, Iterator
from particula.gas.species import GasSpecies
from particula.gas.atmosphere import Atmosphere
from particula.particles.representation import ParticleRepresentation


class Aerosol:
    """Collection of Gas and Particle objects.

    A class for interacting with collections of Gas and Particle objects.
    Allows for the representation and manipulation of an aerosol, which
    is composed of various gases and particles.
    """

    def __init__(
        self,
        atmosphere: Atmosphere,
        particles: Union[ParticleRepresentation, List[ParticleRepresentation]],
    ):
        """
        Parameters:
            atmosphere: Atmosphere with many GasSpeices possible.
            particles: One or more Particle instances.
        """
        self.atmosphere = atmosphere
        self.particles: List[ParticleRepresentation] = (
            [particles]
            if isinstance(particles, ParticleRepresentation)
            else particles
        )

    def __str__(self) -> str:
        """Returns a string representation of the aerosol.

        Returns:
            str: A string representation of the aerosol.
        """
        message = str(self.atmosphere)
        for index, particle in enumerate(self.particles):
            message += f"\n[{index}]: {str(particle)}"

        return message

    def iterate_gas(self) -> Iterator[GasSpecies]:
        """Returns an iterator for atmosphere species.

        Returns:
            Iterator[GasSpecies]: An iterator over the gas species type.
        """
        return iter(self.atmosphere)

    def iterate_particle(self) -> Iterator[ParticleRepresentation]:
        """Returns an iterator for particle.

        Returns:
            Iterator[Particle]: An iterator over the particle type.
        """
        return iter(self.particles)

    def replace_atmosphere(self, atmosphere: Atmosphere):
        """Replaces the current Atmosphere instance with a new one.

        Parameters:
            gas: The instance to replace the current one.
        """
        self.atmosphere = atmosphere

    def add_particle(self, particle: ParticleRepresentation):
        """Adds a Particle instance to the aerosol.

        Parameters:
            particle: The Particle instance to add.
        """
        self.particles.append(particle)
