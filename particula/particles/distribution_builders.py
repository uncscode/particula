"""Builds distributions strategies based on the specified representation.

Currently, there are no parameters to set, but this is used for consistency
with other builder patterns in the codebase.
"""

from particula.abc_builder import BuilderABC
from particula.particles.distribution_strategies import (
    MassBasedMovingBin,
    ParticleResolvedSpeciatedMass,
    RadiiBasedMovingBin,
    SpeciatedMassMovingBin,
)


class MassBasedMovingBinBuilder(BuilderABC):
    """Builds and configures a MassBasedMovingBin instance for mass-based
    distributions.

    This builder requires no parameters, but is kept for consistency with
    other builder patterns. Ensures a uniform interface for creating
    MassBasedMovingBin objects.

    Methods:
    - build : Return a MassBasedMovingBin instance.

    Examples:
        ```py title="Example"
        import particula as par

        builder = par.particles.MassBasedMovingBinBuilder()
        strategy = builder.build()
        # strategy -> MassBasedMovingBin()
        ```
    """

    def __init__(self) -> None:
        """Initialize the MassBasedMovingBinBuilder.

        Sets up the builder with no required parameters for creating a
        MassBasedMovingBin strategy instance.
        """
        required_parameters = None
        BuilderABC.__init__(self, required_parameters)

    def build(self) -> MassBasedMovingBin:
        """Build and return a MassBasedMovingBin instance.

        Returns:
            - MassBasedMovingBin : A strategy for mass-based particle
                distributions.

        Examples:
            ```py title="Build Example"
            import particula as par
            builder = par.particles.MassBasedMovingBinBuilder()
            strategy = builder.build()
            ```
        """
        return MassBasedMovingBin()


class RadiiBasedMovingBinBuilder(BuilderABC):
    """Builds and configures a RadiiBasedMovingBin instance for radius-based
    distributions.

    This builder requires no parameters, but is provided for consistency
    with other builder patterns. Ensures a uniform interface for creating
    RadiiBasedMovingBin objects.

    Methods:
    - build : Return a RadiiBasedMovingBin instance.

    Examples:
        ```py title="Example"
        import particula as par
        builder = par.particles.RadiiBasedMovingBinBuilder()
        strategy = builder.build()
        # strategy -> RadiiBasedMovingBin()
        ```
    """

    def __init__(self) -> None:
        """Initialize the RadiiBasedMovingBinBuilder.

        Sets up the builder with no required parameters for creating a
        RadiiBasedMovingBin strategy instance.
        """
        required_parameters = None
        BuilderABC.__init__(self, required_parameters)

    def build(self) -> RadiiBasedMovingBin:
        """Build and return a RadiiBasedMovingBin instance.

        Returns:
            - RadiiBasedMovingBin : A strategy for radius-based particle
                distributions.

        Examples:
            ```py title="Build Example"
            import particula as par
            builder = par.particles.RadiiBasedMovingBinBuilder()
            strategy = builder.build()
            ```
        """
        return RadiiBasedMovingBin()


class SpeciatedMassMovingBinBuilder(BuilderABC):
    """Builds and configures a SpeciatedMassMovingBin instance for speciated
    mass distributions.

    This builder requires no parameters, but provides consistency with
    other builder patterns and ensures a uniform interface for creating
    SpeciatedMassMovingBin objects.

    Methods:
    - build : Return a SpeciatedMassMovingBin instance.

    Examples:
        ```py title="Example"
        import particula as par
        builder = par.particles.SpeciatedMassMovingBinBuilder()
        strategy = builder.build()
        # strategy -> SpeciatedMassMovingBin()
        ```
    """

    def __init__(self) -> None:
        """Initialize the SpeciatedMassMovingBinBuilder.

        Sets up the builder with no required parameters for creating a
        SpeciatedMassMovingBin strategy instance.
        """
        required_parameters = None
        BuilderABC.__init__(self, required_parameters)

    def build(self) -> SpeciatedMassMovingBin:
        """Build and return a SpeciatedMassMovingBin instance.

        Returns:
            - SpeciatedMassMovingBin : A strategy for speciated mass
                distributions.

        Examples:
            ```py title="Build Example"
            builder = SpeciatedMassMovingBinBuilder()
            strategy = builder.build()
            ```
        """
        return SpeciatedMassMovingBin()


class ParticleResolvedSpeciatedMassBuilder(BuilderABC):
    """Builds and configures a ParticleResolvedSpeciatedMass instance.

    This builder requires no parameters, but follows the same pattern
    to ensure uniform usage. ParticleResolvedSpeciatedMass is useful for
    specific calculations when each particle's species composition must
    be resolved individually.

    Methods:
    - build : Return a ParticleResolvedSpeciatedMass instance.

    Examples:
        ```py title="Example"
        import particula as par
        builder = par.particles.ParticleResolvedSpeciatedMassBuilder()
        strategy = builder.build()
        # strategy -> ParticleResolvedSpeciatedMass()
        ```
    """

    def __init__(self) -> None:
        """Initialize the ParticleResolvedSpeciatedMassBuilder.

        Sets up the builder with no required parameters for creating a
        ParticleResolvedSpeciatedMass strategy instance.
        """
        required_parameters = None
        BuilderABC.__init__(self, required_parameters)

    def build(self) -> ParticleResolvedSpeciatedMass:
        """Build and return a ParticleResolvedSpeciatedMass instance.

        Returns:
            - ParticleResolvedSpeciatedMass : A strategy that resolves
              each particle's species composition independently.

        Examples:
            ```py title="Build Example"
            import particula as par
            builder = par.particles.ParticleResolvedSpeciatedMassBuilder()
            strategy = builder.build()
            ```
        """
        return ParticleResolvedSpeciatedMass()
