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
        self._replace_data(
            particles=particles, gas=gas, environment=environment
        )

    @property
    def particles(self) -> ParticleData:
        """Return the held particle container by identity."""
        return self._particles

    @particles.setter
    def particles(self, particles: ParticleData) -> None:
        """Validate and replace particles, retaining the other two objects."""
        self._replace_data(
            particles=particles, gas=self.gas, environment=self.environment
        )

    @property
    def gas(self) -> GasData:
        """Return the held full gas container by identity."""
        return self._gas

    @gas.setter
    def gas(self, gas: GasData) -> None:
        """Validate and replace gas without changing the held peers."""
        self._replace_data(
            particles=self.particles, gas=gas, environment=self.environment
        )

    @property
    def environment(self) -> EnvironmentData:
        """Return the held environment container by identity."""
        return self._environment

    @environment.setter
    def environment(self, environment: EnvironmentData) -> None:
        """Validate and replace environment without changing the held peers."""
        self._replace_data(
            particles=self.particles, gas=self.gas, environment=environment
        )

    def replace_data(
        self,
        *,
        particles: ParticleData,
        gas: GasData,
        environment: EnvironmentData,
    ) -> None:
        """Publish a complete candidate triple only after read-only admission.

        Args:
            particles: Tagged CPU particle data, retained by identity.
            gas: Full ordered gas data, retained by identity.
            environment: Matching per-box environment, retained by identity.

        Raises:
            TypeError: If a supplied container has the wrong type.
            ValueError: If structure or particle representation is invalid.

        Rejection leaves old and candidate state unchanged. Same-object
        replacements revalidate current data. Success selects supplied state;
        it neither conserves/rescales inventory nor rebinds GPU resources.
        This is validation atomicity, not a concurrent-access guarantee.
        """
        self._replace_data(
            particles=particles, gas=gas, environment=environment
        )

    def _replace_data(
        self,
        *,
        particles: ParticleData,
        gas: GasData,
        environment: EnvironmentData,
    ) -> None:
        """Validate candidates before publishing their references."""
        validate_aerosol_structure(particles, gas, environment)
        particles.validate_representation()
        self._particles, self._gas, self._environment = (
            particles,
            gas,
            environment,
        )

    def __str__(self) -> str:
        """Summarize the current three containers."""
        return f"{self.particles}\n{self.gas}\n{self.environment}"
