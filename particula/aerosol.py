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
    """
    Represents a collection of Gas and Particle objects forming an aerosol
    environment.

    This class allows for the representation and manipulation of an aerosol,
    which consists of various gases in an Atmosphere object and one or more
    ParticleRepresentation objects.

    Attributes:
        - atmosphere : The atmosphere containing the gases.
        - particles : A list of particles in the aerosol.

    Methods:
        - iterate_gas: Returns an iterator over the gas species in atmosphere.
        - iterate_particle: Returns an iterator over ParticleRepresentation
            objects.
        - replace_atmosphere: Replaces the current atmosphere with a new one.
        - add_particle: Adds a new ParticleRepresentation object to the
            aerosol.

    Examples:
        ```py title="Creating an Aerosol"
        aerosol_instance = Aerosol(atmosphere, particles)
        print(aerosol_instance)
        ```

        ```py title="Iterating over the Aerosol"
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
        Initialize the Aerosol object with an Atmosphere and one or more
        particles.

        Arguments:
            - atmosphere : Atmosphere containing one or more gas species.
            - particles : A single ParticleRepresentation or a list of them.
        """
        self.atmosphere = atmosphere
        self.particles: List[ParticleRepresentation] = (
            [particles]
            if isinstance(particles, ParticleRepresentation)
            else particles
        )

    def __str__(self) -> str:
        """
        Provide a string representation of the aerosol.

        Returns:
            - str : A string summarizing the atmosphere and each particle.

        Examples:
            ```py
            aerosol_instance = Aerosol(atmosphere, particles)
            print(aerosol_instance)
            ```
        """
        message = str(self.atmosphere)
        for index, particle in enumerate(self.particles):
            message += f"\n[{index}]: {str(particle)}"

        return message

    def iterate_gas(self) -> Iterator[GasSpecies]:
        """
        Return an iterator over the gas species in the atmosphere.

        Returns:
        - Iterator[GasSpecies] : An iterator over gas species objects.

        Examples:
            ```py title="Iterating over aerosol gas"
            aerosol_instance = Aerosol(atmosphere, particles)
            for gas in aerosol_instance.iterate_gas():
                print(gas)
            ```
        """
        return iter(self.atmosphere)

    def iterate_particle(self) -> Iterator[ParticleRepresentation]:
        """
        Return an iterator over the particle representations.

        Returns:
            - Iterator[ParticleRepresentation] : An iterator over particle
                objects.

        Examples:
            ```py title="Iterating over aerosol particles"
            aerosol_instance = Aerosol(atmosphere, particles)
            for particle in aerosol_instance.iterate_particle():
                print(particle)
            ```
        """
        return iter(self.particles)

    def replace_atmosphere(self, atmosphere: Atmosphere):
        """
        Replace the current atmosphere with a new Atmosphere instance.

        Arguments:
            - atmosphere : The new Atmosphere to assign.

        Examples:
            ```py title="Replacing the Atmosphere in the Aerosol"
            aerosol_instance = Aerosol(atmosphere, particles)
            new_atmosphere = Atmosphere()
            aerosol_instance.replace_atmosphere(new_atmosphere)
            ```
        """
        self.atmosphere = atmosphere

    def add_particle(self, particle: ParticleRepresentation):
        """
        Add a new ParticleRepresentation to the aerosol.

        Arguments:
            - particle : The particle instance to add.

        Examples:
            ```py title="Adding a Particle to the Aerosol"
            aerosol_instance = Aerosol(atmosphere, particles)
            new_particle = ParticleRepresentation()
            aerosol_instance.add_particle(new_particle)
            ```
        """
        self.particles.append(particle)
