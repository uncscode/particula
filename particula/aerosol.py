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

    Attributes:
        - atmosphere : The atmosphere containing the gases.
        - particles : A list of particles in the aerosol.

    Methods:
        - iterate_gas : Returns an iterator for atmosphere species.
        - iterate_particle : Returns an iterator for particle.
        - replace_atmosphere : Replaces the current Atmosphere instance
            with a new one.
        - add_particle : Adds a Particle instance to the aerosol.

    Examples:
        ``` py title="Creating an Aerosol"
        aerosol_instance = Aerosol(atmosphere, particles)
        print(aerosol_instance)
        ```

        ``` py title="Iterating over the Aerosol"
        aerosol_instance = Aerosol(atmosphere, particles)
        for gas in aerosol_instance.iterate_gas():
            print(gas)
        for particle in aerosol_instance.iterate_particle():
            print(particle)
        ```
    """

    def __init__(
        self,
        atmosphere: Atmosphere,
        particles: Union[ParticleRepresentation, List[ParticleRepresentation]],
    ):
        """
        Parameters:
            - atmosphere : Atmosphere with many GasSpeices possible.
            - particles : One or more Particle instances.
        """
        self.atmosphere = atmosphere
        self.particles: List[ParticleRepresentation] = (
            [particles] if isinstance(particles, ParticleRepresentation) else particles
        )

    def __str__(self) -> str:
        """Returns a string representation of the aerosol.

        Returns:
            - str : A string representation of the aerosol.

        Examples:
            ``` py
            aerosol_instance = Aerosol(atmosphere, particles)
            print(aerosol_instance)
            ```
        """
        message = str(self.atmosphere)
        for index, particle in enumerate(self.particles):
            message += f"\n[{index}]: {str(particle)}"

        return message

    def iterate_gas(self) -> Iterator[GasSpecies]:
        """Returns an iterator for atmosphere species.

        Returns:
            - Iterator : An iterator over the gas species type.

        Examples:
            ``` py title="Iterating over the Aerosol gas"
            aerosol_instance = Aerosol(atmosphere, particles)
            for gas in aerosol_instance.iterate_gas():
                print(gas)
            ```
        """
        return iter(self.atmosphere)

    def iterate_particle(self) -> Iterator[ParticleRepresentation]:
        """Returns an iterator for particle.

        Returns:
            - Iterator : An iterator over the particle type.

        Examples:
            ``` py title="Iterating over the Aerosol particle"
            aerosol_instance = Aerosol(atmosphere, particles)
            for particle in aerosol_instance.iterate_particle():
                print(particle)
            ```
        """
        return iter(self.particles)

    def replace_atmosphere(self, atmosphere: Atmosphere):
        """Replaces the current Atmosphere instance with a new one.

        Parameters:
            - atmosphere : The instance to replace the current one.

        Examples:
            ``` py title="Replacing the Atmosphere in the Aerosol"
            aerosol_instance = Aerosol(atmosphere, particles)
            new_atmosphere = Atmosphere()
            aerosol_instance.replace_atmosphere(new_atmosphere)
            ```
        """
        self.atmosphere = atmosphere

    def add_particle(self, particle: ParticleRepresentation):
        """Adds a Particle instance to the aerosol.

        Parameters:
            - particle : The Particle instance to add.

        Examples:
            ``` py title="Adding a Particle to the Aerosol"
            aerosol_instance = Aerosol(atmosphere, particles)
            new_particle = ParticleRepresentation()
            aerosol_instance.add_particle(new_particle)
            ```
        """
        self.particles.append(particle)
