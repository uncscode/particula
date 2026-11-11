"""Identity-preserving aggregate of native CPU aerosol data."""

from particula.aerosol_validation import validate_aerosol_structure
from particula.gas.environment_data import EnvironmentData
from particula.gas.gas_data import GasData
from particula.particles.particle_data import ParticleData


class Aerosol:
    """Hold particle, gas and environment containers without copying.

    Array mutation remains caller-owned. Processes must revalidate their
    physical and chemical requirements. Construction never transfers data or
    rebinds prepared execution state.
    """

    def __init__(
        self,
        *,
        particles: ParticleData,
        gas: GasData,
        environment: EnvironmentData,
    ) -> None:
        """Validate the complete triple before retaining supplied identities.

        Args:
            particles: Explicitly tagged CPU particle storage.
            gas: Full ordered CPU gas storage, including inert species.
            environment: CPU environment with one ratio lane per gas species.

        Raises:
            TypeError: If a container has the wrong type.
            ValueError: If structure or particle representation is invalid.
        """
        validate_aerosol_structure(particles, gas, environment)
        particles.validate_representation()
        self._particles = particles
        self._gas = gas
        self._environment = environment

    @property
    def particles(self) -> ParticleData:
        """Return the held particle container by identity."""
        return self._particles

    @property
    def gas(self) -> GasData:
        """Return the held full gas container by identity."""
        return self._gas

    @property
    def environment(self) -> EnvironmentData:
        """Return the held environment container by identity."""
        return self._environment

    def __str__(self) -> str:
        """Summarize the current three containers."""
        return f"{self.particles}\n{self.gas}\n{self.environment}"
