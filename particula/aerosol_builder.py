"""Fluent assembly of an identity-preserving native Aerosol."""

import logging
from typing import cast

from particula.abc_builder import BuilderABC
from particula.aerosol import Aerosol
from particula.gas import EnvironmentData, GasData
from particula.particles.particle_data import ParticleData

logger = logging.getLogger("particula")


class AerosolBuilder(BuilderABC):
    """Retain three supplied CPU containers and validate them at build time.

    No default containers, copies, species filtering, or physics configuration
    are created. Repeated builds share the supplied data by identity.
    """

    def __init__(self) -> None:
        """Require particles, gas, and environment for every build."""
        super().__init__(["particles", "gas", "environment"])
        self.particles: ParticleData | None = None
        self.gas: GasData | None = None
        self.environment: EnvironmentData | None = None

    def set_particles(
        self, particles: ParticleData, particles_units: str | None = None
    ) -> "AerosolBuilder":
        """Retain particles; optional builder-protocol units are ignored."""
        if particles_units is not None:
            logger.warning("Ignoring units for particles parameter.")
        self.particles = particles
        return self

    def set_gas(
        self, gas: GasData, gas_units: str | None = None
    ) -> "AerosolBuilder":
        """Retain full gas data; optional builder-protocol units are ignored."""
        if gas_units is not None:
            logger.warning("Ignoring units for gas parameter.")
        self.gas = gas
        return self

    def set_environment(
        self,
        environment: EnvironmentData,
        environment_units: str | None = None,
    ) -> "AerosolBuilder":
        """Retain environment; optional builder-protocol units are ignored."""
        if environment_units is not None:
            logger.warning("Ignoring units for environment parameter.")
        self.environment = environment
        return self

    def build(self) -> Aerosol:
        """Construct a fresh aggregate holding the exact supplied containers.

        Returns:
            Native Aerosol after structural and representation admission.

        Raises:
            ValueError: If any component is missing or invalid.
            TypeError: If a supplied component has the wrong container type.
        """
        self.pre_build_check()
        return Aerosol(
            particles=cast(ParticleData, self.particles),
            gas=cast(GasData, self.gas),
            environment=cast(EnvironmentData, self.environment),
        )
