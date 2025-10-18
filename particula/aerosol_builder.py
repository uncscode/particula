"""AerosolBuilder Module.

Provides a fluent interface to create an `Aerosol` instance from an
`Atmosphere` and a `ParticleRepresentation`.  The builder validates
consistency between the gas and particle phases (e.g., the number of
partitioning species) before constructing the final object.

Examples:
```py title="Typical Usage"
from particula.aerosol_builder import AerosolBuilder
aerosol = (
    AerosolBuilder()
    .set_atmosphere(my_atmosphere)
    .set_particles(my_particles)
    .build()
)
```
"""

import logging
from typing import Optional

from particula.abc_builder import BuilderABC
from particula.aerosol import Aerosol
from particula.gas.atmosphere import Atmosphere
from particula.particles.representation import ParticleRepresentation

logger = logging.getLogger("particula")


class AerosolBuilder(BuilderABC):
    """Fluent builder for `Aerosol` objects.

    The builder collects the required components—`Atmosphere` and
    `ParticleRepresentation`—and validates that they are mutually
    consistent before instantiating an `Aerosol`.

    Attributes:
        - atmosphere : Working copy of the atmosphere to embed in the aerosol.
        - particles : Working copy of the particle representation.

    Methods:
    - set_atmosphere : Assign the atmosphere component.
    - set_particles  : Assign the particle component.
    - build          : Validate and return a fully-formed `Aerosol`.

    Examples:
        ```py title="Builder Pattern"
        builder = AerosolBuilder()
        builder.set_atmosphere(atm).set_particles(prt)
        aerosol = builder.build()
        ```
    """

    def __init__(self):
        """Initialize an empty builder.

        Returns:
        - None
        """
        required_parameters = ["atmosphere", "particles"]
        BuilderABC.__init__(self, required_parameters)
        self.atmosphere: Optional[Atmosphere] = None
        self.particles: Optional[ParticleRepresentation] = None

    def set_atmosphere(
        self, atmosphere: Atmosphere, atmosphere_units: Optional[str] = None
    ) -> "AerosolBuilder":
        """Attach an `Atmosphere` to the builder.

        Args:
            atmosphere: Atmosphere to embed in the aerosol.
            atmosphere_units: Ignored; included for signature uniformity.

        Returns:
            AerosolBuilder: The builder instance (for chaining).

        Examples:
            ```py
            builder = AerosolBuilder().set_atmosphere(atm)
            ```
        """
        if atmosphere_units is not None:
            logger.warning("Ignoring units for atmosphere parameter.")
        self.atmosphere = atmosphere
        return self

    def set_particles(
        self,
        particles: ParticleRepresentation,
        particles_units: Optional[str] = None,
    ) -> "AerosolBuilder":
        """Attach a `ParticleRepresentation` to the builder.

        Args:
            particles: Particle representation to embed in the aerosol.
            particles_units: Ignored; included for signature uniformity.

        Returns:
            The builder instance (for chaining).
        """
        if particles_units is not None:
            logger.warning("Ignoring units for particles parameter.")
        self.particles = particles
        return self

    def _validate_species_length(self) -> None:
        """Validate species-count consistency between atmosphere and particles.

        When the particle distribution strategy is either
        `SpeciatedMassMovingBin` or `ParticleResolvedSpeciatedMass`, the number
        of partitioning gas species must equal the number of particle-phase
        species.

        Raises:
            - ValueError : If the species counts are inconsistent.
        """
        if self.atmosphere is None or self.particles is None:
            raise ValueError(
                "Atmosphere and particles must be set before validation."
            )
        strategy_name = self.particles.get_strategy_name()
        if strategy_name not in (
            "SpeciatedMassMovingBin",
            "ParticleResolvedSpeciatedMass",
        ):
            return  # no check required for other strategies

        partitioning_species = self.atmosphere.partitioning_species
        if partitioning_species is None:
            return  # nothing to validate

        # count partitioning species in the atmosphere
        n_partitioning = len(partitioning_species)

        # count species represented by the particle model
        species_mass = self.particles.get_species_mass(clone=False)
        n_particle_species = (
            1 if species_mass.ndim == 1 else species_mass.shape[1]
        )

        if n_partitioning != n_particle_species:
            msg = (
                "Number of partitioning species in atmosphere "
                f"({n_partitioning}) must match number of species in "
                f"particle representation ({n_particle_species}) when using "
                f"{strategy_name}."
            )
            logger.error(msg)
            raise ValueError(msg)

    def build(self) -> Aerosol:
        """Finalize and construct an `Aerosol` object.

        Raises:
            - ValueError : If either component is missing or validation fails.

        Returns:
            Aerosol object, The validated aerosol instance.

        Examples:
        ```py
        aerosol = (
            AerosolBuilder()
            .set_atmosphere(atm)
            .set_particles(prt)
            .build()
        )
        ```
        """
        self.pre_build_check()
        self._validate_species_length()
        return Aerosol(atmosphere=self.atmosphere, particles=self.particles)  # type: ignore
