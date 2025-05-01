"""
Builds the aerosol object from the given parameters.

Apply validation of matching parameters between the atmosphere
and the particles.
"""

import logging
from typing import Optional
import numpy as np

from particula.abc_builder import BuilderABC
from particula.aerosol import Aerosol
from particula.gas.atmosphere import Atmosphere
from particula.gas.species import GasSpecies
from particula.particles.representation import ParticleRepresentation

logger = logging.getLogger("particula")


class AerosolBuilder(BuilderABC):
    def __init__(self):
        required_parameters = ["atmosphere", "particles"]
        BuilderABC.__init__(self, required_parameters)
        self.atmosphere: Optional[Atmosphere] = None
        self.particles: Optional[ParticleRepresentation] = None

    def set_atmosphere(
        self, atmosphere: Atmosphere, atmosphere_units: Optional[str] = None
    ) -> "AerosolBuilder":
        if atmosphere_units is not None:
            logger.warning("Ignoring units for atmosphere parameter.")
        self.atmosphere = atmosphere
        return self

    def set_particles(
        self,
        particles: ParticleRepresentation,
        particles_units: Optional[str] = None,
    ) -> "AerosolBuilder":
        if particles_units is not None:
            logger.warning("Ignoring units for particles parameter.")
        self.particles = particles
        return self

    def build(self) -> Aerosol:
        self.pre_build_check()

        # --- validation of partitioning species count -----------------
        partitioning_species = self.atmosphere.partitioning_species

        if partitioning_species is not None:
            # count partitioning species
            if isinstance(partitioning_species, GasSpecies):
                n_partitioning = 1
            else:
                n_partitioning = len(partitioning_species)

            # count species represented in particles
            try:
                species_mass = self.particles.get_species_mass(clone=False)
                n_particle_species = (
                    1 if species_mass.ndim == 1 else species_mass.shape[0]
                )
            except AttributeError:
                density = self.particles.get_density(clone=False)
                if isinstance(density, np.ndarray) and density.ndim > 0:
                    n_particle_species = density.shape[0]
                else:
                    n_particle_species = 1

            if n_partitioning != n_particle_species:
                message = (
                    "Number of partitioning species in atmosphere "
                    f"({n_partitioning}) must match number of species in "
                    f"particle representation ({n_particle_species})."
                )
                logger.error(message)
                raise ValueError(message)

        return Aerosol(atmosphere=self.atmosphere, particles=self.particles)
