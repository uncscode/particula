"""Aerosol class just a collection of atmosphere (gas species) and particles.

Used to pass state information to the dynamics solvers.
"""

from particula.gas.atmosphere import Atmosphere
from particula.particles.representation import ParticleRepresentation


class Aerosol:
    """Aerosol is a collection of Gas and Particle objects.

    This class allows for the representation and manipulation of an aerosol,
    which consists of various gases in an Atmosphere object and one
    ParticleRepresentation object.

    Attributes:
        - atmosphere : The atmosphere containing the gases.
        - particles : The particle Representation object.

    Methods:
    - replace_atmosphere: Replaces the current atmosphere with a new one.
    - replace_particles: Replaces a particle in the aerosol with a new one.

    Examples:
        ```py title="Creating an Aerosol"
        aerosol_instance = Aerosol(atmosphere, particles)
        print(aerosol_instance)
        ```

        ```py title="replace a particle object"
        aerosol_instance = Aerosol(atmosphere, particles)
        aerosol_instance.replace_particles(new_particle)
        ```
    """

    def __init__(
        self,
        atmosphere: Atmosphere,
        particles: ParticleRepresentation,
    ):
        """Initialize the Aerosol object with an Atmosphere and Particles.

        Args:
            atmosphere: Atmosphere containing one or more gas species.
            particles: A single ParticleRepresentation object.
        """
        self.atmosphere = atmosphere
        self.particles = particles

    def __str__(self) -> str:
        """Provide a string representation of the aerosol.

        Returns:
            - str : A string summarizing the atmosphere and each particle.

            ```py
            aerosol_instance = Aerosol(atmosphere, particles)
            print(aerosol_instance)
            ```
        """
        message = str(self.atmosphere) + "\n" + str(self.particles)
        return message

    def replace_atmosphere(self, atmosphere: Atmosphere):
        """Replace the current atmosphere with a new Atmosphere instance.

        Args:
            atmosphere: The new Atmosphere to assign.

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
        """Replace a particles in the aerosol with a new ParticleRepresentation.

        Args:
            particles: The new ParticleRepresentation to assign.

        Examples:
            ```py title="Replacing a Particle in the Aerosol"
            aerosol_instance = Aerosol(atmosphere, particles)
            new_particle = ParticleRepresentation()
            aerosol_instance.replace_particles(new_particle)
            ```
        """
        self.particles = particles
