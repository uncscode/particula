"""Aerosol class just a list of gas classes and particle classes.

There is a problem here, with matching of gases that can condense to,
particles and getting it to work correctly. This is will be solved
with usage as we figure out the best way to do this.
"""

from typing import Iterator
from particula.gas.species import GasSpecies
from particula.gas.atmosphere import Atmosphere
from particula.particles.representation import ParticleRepresentation


class Aerosol:
    """
    Represents a collection of Gas and Particle objects forming an aerosol
    environment.

    This class allows for the representation and manipulation of an aerosol,
    which consists of various gases in an Atmosphere object and one
    ParticleRepresentation object.

    Attributes:
        - atmosphere : The atmosphere containing the gases.
        - particles : The particle Representation object.

    Methods:
    - iterate_gas: Returns an iterator over the gas species in atmosphere.
    - replace_atmosphere: Replaces the current atmosphere with a new one.
    - replace_particle: Replaces a particle in the aerosol with a new one.

    Examples:
        ```py title="Creating an Aerosol"
        aerosol_instance = Aerosol(atmosphere, particles)
        print(aerosol_instance)
        ```

        ```py title="Iterating over the Aerosol"
        aerosol_instance = Aerosol(atmosphere, particles)
        for gas in aerosol_instance.iterate_gas():
            print(gas)
        ```
    """

    def __init__(
        self,
        atmosphere: Atmosphere,
        particles: ParticleRepresentation,
    ):
        """
        Initialize the Aerosol object with an Atmosphere and one or more
        particles.

        Arguments:
            - atmosphere : Atmosphere containing one or more gas species.
            - particles : A single ParticleRepresentation object.
        """
        self.atmosphere = atmosphere
        self.particles = particles

    def __str__(self) -> str:
        """
        Provide a string representation of the aerosol.

        Returns:
            - str : A string summarizing the atmosphere and each particle.

            ```py
            aerosol_instance = Aerosol(atmosphere, particles)
            print(aerosol_instance)
            ```
        """
        message = str(self.atmosphere) + "\n" + str(self.particles)
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

    def replace_particles(
        self,
        particles: ParticleRepresentation,
    ):
        """
        Replace a particles in the aerosol with a new ParticleRepresentation.

        Arguments:
            - particle : The new ParticleRepresentation to assign.

        Examples:
            ```py title="Replacing a Particle in the Aerosol"
            aerosol_instance = Aerosol(atmosphere, particles)
            new_particle = ParticleRepresentation()
            aerosol_instance.replace_particles(new_particle)
            ```
        """
        self.particles = particles
