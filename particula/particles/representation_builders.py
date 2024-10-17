"""
This module contains the builders for the particle representation classes.

Classes:
- MassParticleRepresentationBuilder: Builder class for
    HistogramParticleRepresentation objects.
- PDFParticleRepresentationBuilder: Builder class for PDFParticleRepresentation
    objects.
- DiscreteParticleRepresentationBuilder: Builder class for
    DiscreteParticleRepresentation objects.
"""

from typing import Optional
import logging
from numpy.typing import NDArray
import numpy as np

from particula.util.input_handling import convert_units  # type: ignore

from particula.abc_builder import (
    BuilderABC,
)
from particula.builder_mixin import (
    BuilderRadiusMixin,
    BuilderConcentrationMixin,
    BuilderMassMixin,
    BuilderDensityMixin,
    BuilderChargeMixin,
    BuilderVolumeMixin,
    BuilderLognormalMixin,
    BuilderParticleResolvedCountMixin,
)
from particula.particles.distribution_strategies import (
    DistributionStrategy,
    RadiiBasedMovingBin,
    ParticleResolvedSpeciatedMass,
)
from particula.particles.activity_strategies import (
    ActivityStrategy,
    ActivityIdealMass,
)
from particula.particles.surface_strategies import (
    SurfaceStrategy,
    SurfaceStrategyVolume,
)
from particula.particles.representation import ParticleRepresentation
from particula.particles.properties.lognormal_size_distribution import (
    lognormal_pdf_distribution,
    lognormal_pmf_distribution,
    lognormal_sample_distribution,
)

logger = logging.getLogger("particula")


# pylint: disable=too-few-public-methods
class BuilderSurfaceStrategyMixin:
    """Mixin class for Builder classes to set surface_strategy.

    Methods:
        set_surface_strategy: Set the surface_strategy attribute.
    """

    def __init__(self):
        self.surface_strategy = None

    def set_surface_strategy(
        self,
        surface_strategy: SurfaceStrategy,
        surface_strategy_units: Optional[str] = None,
    ):
        """Set the surface strategy of the particle.

        Args:
            surface_strategy: Surface strategy of the particle.
            surface_strategy_units: Not used. (for interface consistency)
        """
        if surface_strategy_units is not None:
            logger.warning("Ignoring units for surface strategy parameter.")
        self.surface_strategy = surface_strategy
        return self


# pylint: disable=too-few-public-methods
class BuilderDistributionStrategyMixin:
    """Mixin class for Builder classes to set distribution_strategy.

    Methods:
        set_distribution_strategy: Set the distribution_strategy attribute.
    """

    def __init__(self):
        self.distribution_strategy = None

    def set_distribution_strategy(
        self,
        distribution_strategy: DistributionStrategy,
        distribution_strategy_units: Optional[str] = None,
    ):
        """Set the distribution strategy of the particle.

        Args:
            distribution_strategy: Distribution strategy of the particle.
            distribution_strategy_units: Not used. (for interface consistency)
        """
        if distribution_strategy_units is not None:
            logger.warning(
                "Ignoring units for distribution strategy parameter."
            )
        self.distribution_strategy = distribution_strategy
        return self


# pylint: disable=too-few-public-methods
class BuilderActivityStrategyMixin:
    """Mixin class for Builder classes to set activity_strategy.

    Methods:
        set_activity_strategy: Set the activity_strategy attribute.
    """

    def __init__(self):
        self.activity_strategy = None

    def set_activity_strategy(
        self,
        activity_strategy: ActivityStrategy,
        activity_strategy_units: Optional[str] = None,
    ):
        """Set the activity strategy of the particle.

        Args:
            activity_strategy: Activity strategy of the particle.
            activity_strategy_units: Not used. (for interface consistency)
        """
        if activity_strategy_units is not None:
            logger.warning("Ignoring units for activity strategy parameter.")
        self.activity_strategy = activity_strategy
        return self


class ParticleMassRepresentationBuilder(
    BuilderABC,
    BuilderDistributionStrategyMixin,
    BuilderActivityStrategyMixin,
    BuilderSurfaceStrategyMixin,
    BuilderMassMixin,
    BuilderDensityMixin,
    BuilderConcentrationMixin,
    BuilderChargeMixin,
):  # pylint: disable=too-many-ancestors
    """General ParticleRepresentation objects with mass-based bins.

    Attributes:
        distribution_strategy: Set the DistributionStrategy.
        activity_strategy: Set the ActivityStrategy.
        surface_strategy: Set the SurfaceStrategy.
        mass: Set the mass of the particles. Default units are 'kg'.
        density: Set the density of the particles. Default units are 'kg/m^3'.
        concentration: Set the concentration of the particles.
            Default units are '1/m^3'.
        charge: Set the number of charges.
    """

    def __init__(self):
        required_parameters = [
            "distribution_strategy",
            "activity_strategy",
            "surface_strategy",
            "mass",
            "density",
            "concentration",
            "charge",
        ]
        BuilderABC.__init__(self, required_parameters)
        BuilderDistributionStrategyMixin.__init__(self)
        BuilderActivityStrategyMixin.__init__(self)
        BuilderSurfaceStrategyMixin.__init__(self)
        BuilderMassMixin.__init__(self)
        BuilderDensityMixin.__init__(self)
        BuilderConcentrationMixin.__init__(self, default_units="1/m^3")
        BuilderChargeMixin.__init__(self)

    def build(self) -> ParticleRepresentation:
        """Validate and return the ParticleRepresentation object.

        Returns:
            The validated ParticleRepresentation object.
        """
        self.pre_build_check()
        return ParticleRepresentation(
            strategy=self.distribution_strategy,  # type: ignore
            activity=self.activity_strategy,  # type: ignore
            surface=self.surface_strategy,  # type: ignore
            distribution=self.mass,  # type: ignore
            density=self.density,  # type: ignore
            concentration=self.concentration,  # type: ignore
            charge=self.charge,  # type: ignore
        )


class ParticleRadiusRepresentationBuilder(
    BuilderABC,
    BuilderDistributionStrategyMixin,
    BuilderActivityStrategyMixin,
    BuilderSurfaceStrategyMixin,
    BuilderRadiusMixin,
    BuilderDensityMixin,
    BuilderConcentrationMixin,
    BuilderChargeMixin,
):  # pylint: disable=too-many-ancestors
    """General ParticleRepresentation objects with radius-based bins.

    Attributes:
        distribution_strategy: Set the DistributionStrategy.
        activity_strategy: Set the ActivityStrategy.
        surface_strategy: Set the SurfaceStrategy.
        radius: Set the radius of the particles. Default units are 'm'.
        density: Set the density of the particles. Default units are 'kg/m**3'.
        concentration: Set the concentration of the particles. Default units
            are '1/m^3'.
        charge: Set the number of charges.
    """

    def __init__(self):
        required_parameters = [
            "distribution_strategy",
            "activity_strategy",
            "surface_strategy",
            "radius",
            "density",
            "concentration",
            "charge",
        ]
        BuilderABC.__init__(self, required_parameters)
        BuilderDistributionStrategyMixin.__init__(self)
        BuilderActivityStrategyMixin.__init__(self)
        BuilderSurfaceStrategyMixin.__init__(self)
        BuilderRadiusMixin.__init__(self)
        BuilderDensityMixin.__init__(self)
        BuilderConcentrationMixin.__init__(self, default_units="1/m^3")
        BuilderChargeMixin.__init__(self)

    def build(self) -> ParticleRepresentation:
        """Validate and return the ParticleRepresentation object.

        Returns:
            The validated ParticleRepresentation object.
        """
        self.pre_build_check()
        return ParticleRepresentation(
            strategy=self.distribution_strategy,  # type: ignore
            activity=self.activity_strategy,  # type: ignore
            surface=self.surface_strategy,  # type: ignore
            distribution=self.radius,  # type: ignore
            density=self.density,  # type: ignore
            concentration=self.concentration,  # type: ignore
            charge=self.charge,  # type: ignore
        )


class PresetParticleRadiusBuilder(
    BuilderABC,
    BuilderDistributionStrategyMixin,
    BuilderActivityStrategyMixin,
    BuilderSurfaceStrategyMixin,
    BuilderRadiusMixin,
    BuilderDensityMixin,
    BuilderConcentrationMixin,
    BuilderChargeMixin,
    BuilderLognormalMixin,
):  # pylint: disable=too-many-ancestors
    """General ParticleRepresentation objects with radius-based bins.

    Attributes:
        mode: Set the mode(s) of the distribution.
            Default is np.array([100e-9, 1e-6]) meters.
        geometric_standard_deviation: Set the geometric standard deviation(s)
            of the distribution. Default is np.array([1.2, 1.4]).
        number_concentration: Set the number concentration of the distribution.
            Default is np.array([1e4x1e6, 1e3x1e6]) particles/m^3.
        radius_bins: Set the radius bins of the distribution. Default is
            np.logspace(-9, -4, 250), meters.
    """

    def __init__(self):
        required_parameters = [
            "mode",
            "geometric_standard_deviation",
            "number_concentration",
        ]
        BuilderABC.__init__(self, required_parameters)
        BuilderDistributionStrategyMixin.__init__(self)
        BuilderActivityStrategyMixin.__init__(self)
        BuilderSurfaceStrategyMixin.__init__(self)
        BuilderRadiusMixin.__init__(self)
        BuilderDensityMixin.__init__(self)
        BuilderConcentrationMixin.__init__(self, default_units="1/m^3")
        BuilderChargeMixin.__init__(self)
        BuilderLognormalMixin.__init__(self)

        # set defaults
        self.mode = np.array([100e-9, 1e-6])
        self.geometric_standard_deviation = np.array([1.2, 1.4])
        self.number_concentration = np.array([1e4 * 1e6, 1e3 * 1e6])
        self.radius_bins = np.logspace(-9, -4, 250)
        self.set_distribution_strategy(RadiiBasedMovingBin())
        self.set_activity_strategy(ActivityIdealMass())
        self.set_surface_strategy(
            SurfaceStrategyVolume(surface_tension=0.072, density=1000)
        )
        self.set_density(1000, "kg/m^3")
        self.set_charge(0)
        self.distribution_type = "pmf"

    def set_radius_bins(
        self,
        radius_bins: NDArray[np.float64],
        radius_bins_units: str = "m",
    ):
        """Set the radius bins for the distribution

        Arguments:
            radius_bins: The radius bins for the distribution.
        """
        if np.any(radius_bins < 0):
            message = "The radius bins must be positive."
            logger.error(message)
            raise ValueError(message)
        self.radius_bins = radius_bins * convert_units(radius_bins_units, "m")
        return self

    def set_distribution_type(
        self,
        distribution_type: str,
        distribution_type_units: Optional[str] = None,
    ):
        """Set the distribution type for the particle representation.

        Arguments:
            distribution_type: The type of distribution to use.
        """
        if distribution_type not in ["pdf", "pmf"]:
            message = "The distribution type must be either 'pdf' or 'pmf'."
            logger.error(message)
            raise ValueError(message)
        if distribution_type_units is not None:
            logger.warning("Ignoring units for distribution type parameter.")
        self.distribution_type = distribution_type
        return self

    def build(self) -> ParticleRepresentation:
        """Validate and return the ParticleRepresentation object.

        This will build a distribution of particles with a lognormal size
        distribution, before returning the ParticleRepresentation object.

        Returns:
            The validated ParticleRepresentation object.
        """
        if self.distribution_type == "pdf":
            number_concentration = lognormal_pdf_distribution(
                x_values=self.radius_bins,
                mode=self.mode,
                geometric_standard_deviation=self.geometric_standard_deviation,
                number_of_particles=self.number_concentration,
            )
        elif self.distribution_type == "pmf":
            number_concentration = lognormal_pmf_distribution(
                x_values=self.radius_bins,
                mode=self.mode,
                geometric_standard_deviation=self.geometric_standard_deviation,
                number_of_particles=self.number_concentration,
            )
        else:
            message = "The distribution type must be either 'pdf' or 'pmf'."
            logger.error(message)
            raise ValueError(message)

        self.pre_build_check()
        return ParticleRepresentation(
            strategy=self.distribution_strategy,  # type: ignore
            activity=self.activity_strategy,  # type: ignore
            surface=self.surface_strategy,  # type: ignore
            distribution=self.radius_bins,
            density=self.density,  # type: ignore
            concentration=number_concentration,  # type: ignore
            charge=self.charge,  # type: ignore
        )


class ResolvedParticleMassRepresentationBuilder(
    BuilderABC,
    BuilderDistributionStrategyMixin,
    BuilderActivityStrategyMixin,
    BuilderSurfaceStrategyMixin,
    BuilderDensityMixin,
    BuilderChargeMixin,
    BuilderVolumeMixin,
    BuilderMassMixin,
):  # pylint: disable=too-many-ancestors
    """
    Builder class for constructing ParticleRepresentation objects with
    resolved masses.

    This class allows you to set various attributes for a particle
    representation, such as distribution strategy, mass, density, charge,
    volume, and more. These attributes are validated and there a no presets.

    Attributes:
        distribution_strategy: Set the distribution strategy for particles.
        activity_strategy: Set the activity strategy for the particles.
        surface_strategy: Set the surface strategy for the particles.
        mass: Set the particle mass. Defaults to 'kg'.
        density: Set the particle density. Defaults to 'kg/m^3'.
        charge: Set the particle charge.
        volume: Set the particle volume. Defaults to 'm^3'.
    """

    def __init__(self):
        required_parameters = [
            "distribution_strategy",
            "activity_strategy",
            "surface_strategy",
            "mass",
            "density",
            "charge",
            "volume",
        ]
        BuilderABC.__init__(self, required_parameters)
        BuilderDistributionStrategyMixin.__init__(self)
        BuilderActivityStrategyMixin.__init__(self)
        BuilderSurfaceStrategyMixin.__init__(self)
        BuilderMassMixin.__init__(self)
        BuilderDensityMixin.__init__(self)
        BuilderChargeMixin.__init__(self)
        BuilderVolumeMixin.__init__(self)

    def build(self) -> ParticleRepresentation:
        """
        Validate and return a ParticleRepresentation object.

        This method validates all the required attributes and builds a particle
        representation with a lognormal size distribution.

        Returns:
            ParticleRepresentation: A validated particle representation object.
        """
        number_concentration = np.ones_like(  # type: ignore
            self.mass, dtype=np.float64  # type: ignore
        )
        if number_concentration.ndim > 1:
            number_concentration = number_concentration[:, 0]

        self.pre_build_check()
        return ParticleRepresentation(
            strategy=self.distribution_strategy,  # type: ignore
            activity=self.activity_strategy,  # type: ignore
            surface=self.surface_strategy,  # type: ignore
            distribution=self.mass,  # type: ignore
            density=self.density,  # type: ignore
            concentration=number_concentration,  # type: ignore
            charge=self.charge,  # type: ignore
            volume=self.volume,  # type: ignore
        )


class PresetResolvedParticleMassBuilder(
    BuilderABC,
    BuilderDistributionStrategyMixin,
    BuilderActivityStrategyMixin,
    BuilderSurfaceStrategyMixin,
    BuilderDensityMixin,
    BuilderChargeMixin,
    BuilderLognormalMixin,
    BuilderVolumeMixin,
    BuilderParticleResolvedCountMixin,
):  # pylint: disable=too-many-ancestors
    """General ParticleRepresentation objects with particle resolved masses.

    This class has preset values for all the attributes, and allows you to
    override them as needed. This is useful when you want to quickly
    particle representation object with resolved masses.

    Attributes:
        distribution_strategy: Set the DistributionStrategy.
        activity_strategy: Set the ActivityStrategy.
        surface_strategy: Set the SurfaceStrategy.
        mass: Set the mass of the particles Default
            units are 'kg'.
        density: Set the density of the particles.
            Default units are 'kg/m^3'.
        charge: Set the number of charges.
        mode: Set the mode(s) of the distribution.
            Default is np.array([100e-9, 1e-6]) meters.
        geometric_standard_deviation: Set the geometric standard
            deviation(s) of the distribution. Default is np.array([1.2, 1.4]).
        number_concentration: Set the number concentration of the
            distribution. Default is np.array([1e4 1e6, 1e3 1e6])
            particles/m^3.
        particle_resolved_count: Set the number of resolved particles.
    """

    def __init__(self):
        required_parameters = [
            "mode",
            "geometric_standard_deviation",
            "number_concentration",
            "particle_resolved_count",
            "volume",
        ]
        BuilderABC.__init__(self, required_parameters)
        BuilderDistributionStrategyMixin.__init__(self)
        BuilderActivityStrategyMixin.__init__(self)
        BuilderSurfaceStrategyMixin.__init__(self)
        BuilderDensityMixin.__init__(self)
        BuilderChargeMixin.__init__(self)
        BuilderLognormalMixin.__init__(self)
        BuilderVolumeMixin.__init__(self)
        BuilderParticleResolvedCountMixin.__init__(self)

        # set defaults
        self.mode = np.array([100e-9, 1e-6])
        self.geometric_standard_deviation = np.array([1.2, 1.4])
        self.number_concentration = np.array([1e4 * 1e6, 1e3 * 1e6])
        self.particle_resolved_count = int(10_000)
        self.set_distribution_strategy(ParticleResolvedSpeciatedMass())
        self.set_activity_strategy(ActivityIdealMass())
        self.set_surface_strategy(
            SurfaceStrategyVolume(surface_tension=0.072, density=1000)
        )
        self.set_density(1000, "kg/m^3")
        self.set_charge(0)
        self.set_volume(1)

    def build(self) -> ParticleRepresentation:
        """Validate and return the ParticleRepresentation object.

        This will build a distribution of particles with a lognormal size
        distribution, before returning the ParticleRepresentation object.

        Returns:
            The validated ParticleRepresentation object.
        """

        resolved_radii = lognormal_sample_distribution(
            mode=self.mode,
            geometric_standard_deviation=self.geometric_standard_deviation,
            number_of_particles=self.number_concentration,
            number_of_samples=self.particle_resolved_count,
        )
        # convert radii to masses
        resolved_masses = np.float64(
            4 / 3 * np.pi * resolved_radii**3 * self.density  # type: ignore
        )
        number_concentration = np.ones_like(resolved_masses, dtype=np.float64)

        self.pre_build_check()
        return ParticleRepresentation(
            strategy=self.distribution_strategy,  # type: ignore
            activity=self.activity_strategy,  # type: ignore
            surface=self.surface_strategy,  # type: ignore
            distribution=resolved_masses,  # type: ignore
            density=self.density,  # type: ignore
            concentration=number_concentration,
            charge=self.charge,  # type: ignore
            volume=self.volume,  # type: ignore
        )
