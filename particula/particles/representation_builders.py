"""Particle Representation Builders.

Provides builder classes for creating ParticleRepresentation objects
with specialized distribution, activity, and surface strategies for
mass-based, radius-based, discrete, or lognormal representations.
"""

import logging
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from particula.abc_builder import (
    BuilderABC,
)
from particula.builder_mixin import (
    BuilderChargeMixin,
    BuilderConcentrationMixin,
    BuilderDensityMixin,
    BuilderLognormalMixin,
    BuilderMassMixin,
    BuilderParticleResolvedCountMixin,
    BuilderRadiusMixin,
    BuilderVolumeMixin,
)
from particula.particles.activity_strategies import (
    ActivityIdealMass,
    ActivityStrategy,
)
from particula.particles.distribution_strategies import (
    DistributionStrategy,
    ParticleResolvedSpeciatedMass,
    RadiiBasedMovingBin,
)
from particula.particles.properties.lognormal_size_distribution import (
    get_lognormal_pdf_distribution,
    get_lognormal_pmf_distribution,
    get_lognormal_sample_distribution,
)
from particula.particles.representation import ParticleRepresentation
from particula.particles.surface_strategies import (
    SurfaceStrategy,
    SurfaceStrategyVolume,
)
from particula.util.convert_units import get_unit_conversion
from particula.util.validate_inputs import validate_inputs

logger = logging.getLogger("particula")


# pylint: disable=too-few-public-methods
class BuilderSurfaceStrategyMixin:
    """Mixin class for setting the surface_strategy attribute.

    Methods:
    - set_surface_strategy : Assign the surface strategy controlling
        surface tension or other surface-related properties.
    """

    def __init__(self):
        """Initialize the BuilderSurfaceStrategyMixin.

        Sets up the mixin with surface_strategy attribute initialized to
        None, ready for assignment via set_surface_strategy method.
        """
        self.surface_strategy = None

    def set_surface_strategy(
        self,
        surface_strategy: SurfaceStrategy,
        surface_strategy_units: Optional[str] = None,
    ):
        """Assign the surface strategy for the particle representation.

        Arguments:
            - surface_strategy : A SurfaceStrategy instance defining surface
              tension or other surface-related properties.
            - surface_strategy_units : Not used (for interface consistency).

        Returns:
            - self : For method chaining.
        """
        if surface_strategy_units is not None:
            logger.warning("Ignoring units for surface strategy parameter.")
        self.surface_strategy = surface_strategy
        return self


# pylint: disable=too-few-public-methods
class BuilderDistributionStrategyMixin:
    """Mixin class for setting the distribution_strategy attribute.

    Methods:
    - set_distribution_strategy : Assign the distribution strategy
        (e.g., mass-based, radii-based).
    """

    def __init__(self):
        """Initialize the BuilderDistributionStrategyMixin.

        Sets up the mixin with distribution_strategy attribute initialized
        to None, ready for assignment via set_distribution_strategy method.
        """
        self.distribution_strategy = None

    def set_distribution_strategy(
        self,
        distribution_strategy: DistributionStrategy,
        distribution_strategy_units: Optional[str] = None,
    ):
        """Assign the distribution strategy for the particle representation.

        Arguments:
            - distribution_strategy : A DistributionStrategy instance
              (e.g., mass-based bins, radius-based bins).
            - distribution_strategy_units : Not used
                (for interface consistency).

        Returns:
            - self : For method chaining.
        """
        if distribution_strategy_units is not None:
            logger.warning(
                "Ignoring units for distribution strategy parameter."
            )
        self.distribution_strategy = distribution_strategy
        return self


# pylint: disable=too-few-public-methods
class BuilderActivityStrategyMixin:
    """Mixin class for setting the activity_strategy attribute.

    Methods:
    - set_activity_strategy : Assign the activity strategy (e.g., ideal mass,
        ideal molar, kappa-parameter).
    """

    def __init__(self):
        """Initialize the BuilderActivityStrategyMixin.

        Sets up the mixin with activity_strategy attribute initialized to
        None, ready for assignment via set_activity_strategy method.
        """
        self.activity_strategy = None

    def set_activity_strategy(
        self,
        activity_strategy: ActivityStrategy,
        activity_strategy_units: Optional[str] = None,
    ):
        """Assign the activity strategy for the particle representation.

        Arguments:
            - activity_strategy : An ActivityStrategy instance (e.g.,
              ActivityIdealMass, ActivityIdealMolar).
            - activity_strategy_units : Not used (for interface consistency).

        Returns:
            - self : For method chaining.
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
    """Builder for ParticleRepresentation using mass distributions.

    Attributes:
        - distribution_strategy : The DistributionStrategy
            (e.g., mass-based bins).
        - activity_strategy : The ActivityStrategy (e.g., ideal mass).
        - surface_strategy : The SurfaceStrategy
            (e.g., surface tension calculations).
        - mass : The total or per-bin mass of particles in kg.
        - density : The particle density in kg/m^3.
        - concentration : The number concentration in 1/m^3.
        - charge : Number of charges per particle (dimensionless).

    Methods:
    - set_distribution_strategy : Assign the distribution strategy.
    - set_activity_strategy : Assign the activity strategy.
    - set_surface_strategy : Assign the surface strategy.
    - set_mass : Assign the mass of the particles.
    - set_density : Assign the density of the particles.
    - set_concentration : Assign the number concentration.
    - set_charge : Assign the charge of the particles.
    - build : Validate and return a ParticleRepresentation with
        mass-based distribution data.
    """

    def __init__(self):
        """Initialize the ParticleMassRepresentationBuilder.

        Sets up the builder with required parameters for creating a
        ParticleRepresentation using mass-based distributions.
        """
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
        """Validate all required parameters and return a ParticleRepresentation.

        Returns:
            - ParticleRepresentation : An object configured to represent
              mass-based particle distributions, activity, and surface
              properties.
        """
        self.pre_build_check()
        return ParticleRepresentation(
            strategy=self.distribution_strategy,
            activity=self.activity_strategy,
            surface=self.surface_strategy,
            distribution=self.mass,
            density=self.density,
            concentration=self.concentration,  # type: ignore[arg-type]
            charge=self.charge,
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
    """Builder for ParticleRepresentation objects using radius-based
    distributions.

    Attributes:
        - distribution_strategy : The DistributionStrategy (e.g.,
            radius-based bins).
        - activity_strategy : The ActivityStrategy (e.g., ideal mass).
        - surface_strategy : The SurfaceStrategy (e.g., surface tension
            calculations).
        - radius : The radius of the particles in meters.
        - density : The particle density in kg/m^3.
        - concentration : The number concentration in 1/m^3.
        - charge : Number of charges per particle (dimensionless).

    Methods:
    - set_distribution_strategy : Assign the distribution strategy.
    - set_activity_strategy : Assign the activity strategy.
    - set_surface_strategy : Assign the surface strategy.
    - set_radius : Assign the radius of the particles.
    - set_density : Assign the density of the particles.
    - set_concentration : Assign the number concentration.
    - set_charge : Assign the charge of the particles.
    - build : Validate and return a ParticleRepresentation with
      radius-based distribution data.
    """

    def __init__(self):
        """Initialize the ParticleRadiusRepresentationBuilder.

        Sets up the builder with required parameters for creating a
        ParticleRepresentation using radius-based distributions.
        """
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
        """Validate all required parameters and return a ParticleRepresentation.

        Returns:
            - ParticleRepresentation : An object configured to represent
              radius-based distributions, activity, and surface properties.
        """
        self.pre_build_check()
        return ParticleRepresentation(
            strategy=self.distribution_strategy,
            activity=self.activity_strategy,
            surface=self.surface_strategy,
            distribution=self.radius,
            density=self.density,
            concentration=self.concentration,  # type: ignore[arg-type]
            charge=self.charge,
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
    """Builder for ParticleRepresentation objects with radius-based bins
    generated from a lognormal size distribution.

    Attributes:
        - mode : Mode(s) of the lognormal distribution in meters.
        - geometric_standard_deviation : Geometric standard deviation(s).
        - number_concentration : Number concentration(s) in 1/m^3.
        - radius_bins : The array of radius bins in meters for the
            distribution.
        - distribution_type : The type of lognormal distribution
            ("pdf" or "pmf").

    Methods:
    - set_distribution_strategy : Assign the distribution strategy.
    - set_activity_strategy : Assign the activity strategy.
    - set_surface_strategy : Assign the surface strategy.
    - set_radius_bins : Assign radius bin edges in meters.
    - set_distribution_type : Choose between "pdf" or "pmf".
    - build : Generate the distribution and return a ParticleRepresentation.
    """

    def __init__(self):
        """Initialize the PresetParticleRadiusBuilder.

        Sets up the builder with default lognormal distribution parameters
        and default strategies for creating radius-based particle
        representations.
        """
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
        self.set_charge(np.zeros_like(self.radius_bins))
        self.distribution_type = "pmf"

    @validate_inputs({"radius_bins": "positive"})
    def set_radius_bins(
        self,
        radius_bins: NDArray[np.float64],
        radius_bins_units: str = "m",
    ):
        """Assign an array of radius bin edges.

        Arguments:
            - radius_bins : The radius bin edges in meters.
            - radius_bins_units : The units of the radius bins. Default is "m".

        Returns:
            - self : For method chaining.
        """
        if np.any(radius_bins < 0):
            message = "The radius bins must be positive."
            logger.error(message)
            raise ValueError(message)
        if radius_bins_units == "m":
            self.radius_bins = radius_bins
            return self
        self.radius_bins = radius_bins * get_unit_conversion(
            radius_bins_units, "m"
        )
        return self

    def set_distribution_type(
        self,
        distribution_type: str,
        distribution_type_units: Optional[str] = None,
    ):
        """Choose between "pdf" (probability density function) or "pmf"
        (probability mass function) for the distribution.

        Arguments:
            - distribution_type : Must be either 'pdf' or 'pmf'.
            - distribution_type_units : Not used (for interface consistency).

        Returns:
            - self : For method chaining.
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
        """Generate a lognormal distribution (PDF or PMF) based on
        current parameters and return a ParticleRepresentation.

        Returns:
            - ParticleRepresentation : An object with radius-based lognormal
              distribution, activity, and surface properties.
        """
        if self.distribution_type == "pdf":
            number_concentration = get_lognormal_pdf_distribution(
                x_values=self.radius_bins,
                mode=self.mode,
                geometric_standard_deviation=self.geometric_standard_deviation,
                number_of_particles=self.number_concentration,
            )
        elif self.distribution_type == "pmf":
            number_concentration = get_lognormal_pmf_distribution(
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
            strategy=self.distribution_strategy,
            activity=self.activity_strategy,
            surface=self.surface_strategy,
            distribution=self.radius_bins,
            density=self.density,
            concentration=number_concentration,
            charge=self.charge,
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
    """Builder for ParticleRepresentation objects with fully resolved
    particle masses (array-based).

    Allows setting distribution strategy, mass, density, charge,
    volume, etc. No default values are assumed.

    Attributes:
        - distribution_strategy : Strategy for the resolved mass distribution.
        - activity_strategy : Activity strategy (e.g., ideal mass).
        - surface_strategy : Surface strategy (e.g., tension).
        - mass : Per-particle or resolved mass in kg.
        - density : Particle density in kg/m^3.
        - charge : Number of charges (dimensionless).
        - volume : Volume of simulation in m^3.

    Methods:
    - set_distribution_strategy : Assign the distribution strategy.
    - set_activity_strategy : Assign the activity strategy.
    - set_surface_strategy : Assign the surface strategy.
    - set_mass : Assign the mass of the particles.
    - set_density : Assign the density of the particles.
    - set_charge : Assign the charge of the particles.
    - set_volume : Assign the volume of the particles.
    - build : Validate all parameters and return
        a ParticleRepresentation with resolved masses.
    """

    def __init__(self):
        """Initialize the ResolvedParticleMassRepresentationBuilder.

        Sets up the builder with required parameters for creating a
        ParticleRepresentation with fully resolved particle masses.
        """
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
        """Validate attributes and construct a ParticleRepresentation
        with array-based, resolved masses.

        Returns:
            - ParticleRepresentation : Configured with the specified
              distribution, activity, and surface strategies.
        """
        number_concentration = np.ones_like(self.mass, dtype=np.float64)
        if number_concentration.ndim > 1:
            number_concentration = number_concentration[:, 0]

        self.pre_build_check()
        return ParticleRepresentation(
            strategy=self.distribution_strategy,
            activity=self.activity_strategy,
            surface=self.surface_strategy,
            distribution=self.mass,
            density=self.density,
            concentration=number_concentration,
            charge=self.charge,
            volume=self.volume,
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
    """Builder for ParticleRepresentation objects with preset parameters
    for particle-resolved masses derived from a lognormal size distribution.

    Generates random samples of particle radii (lognormal) and converts
    them to per-particle masses. Includes defaults for mode, geometric
    standard deviation, concentration, and total resolved count.

    Attributes:
        - mode : Lognormal mode(s) in meters.
        - geometric_standard_deviation : GSD(s).
        - number_concentration : Number concentration(s) in 1/m^3.
        - particle_resolved_count : Number of resolved particles.
        - volume : Volume in m^3 for the representation.

    Methods:
    - build : Sample radii from a lognormal distribution, convert to mass,
        and create a ParticleRepresentation.
    """

    def __init__(self):
        """Initialize the PresetResolvedParticleMassBuilder.

        Sets up the builder with default lognormal parameters and strategies
        for creating particle-resolved mass representations from sampled
        distributions.
        """
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
        self.particle_resolved_count = int(20_000)
        self.set_distribution_strategy(ParticleResolvedSpeciatedMass())
        self.set_activity_strategy(ActivityIdealMass())
        self.set_surface_strategy(
            SurfaceStrategyVolume(surface_tension=0.072, density=1000)
        )
        self.set_density(1000, "kg/m^3")
        self.set_charge(np.zeros(self.particle_resolved_count))
        self.set_volume(1, "m^3")

    def build(self) -> ParticleRepresentation:
        """Sample particle radii from a lognormal distribution, convert
        to mass, and return a ParticleRepresentation with resolved
        per-particle masses.

        Returns:
            - ParticleRepresentation : Configured for particle-resolved
              masses with the specified distribution, activity, and surface.
        """
        resolved_radii = get_lognormal_sample_distribution(
            mode=self.mode,
            geometric_standard_deviation=self.geometric_standard_deviation,
            number_of_particles=self.number_concentration,
            number_of_samples=self.particle_resolved_count,
        )
        # convert radii to masses
        resolved_masses_calc = 4 / 3 * np.pi * resolved_radii**3 * self.density
        resolved_masses = np.asarray(resolved_masses_calc, dtype=np.float64)
        number_concentration = np.ones_like(resolved_masses, dtype=np.float64)

        self.pre_build_check()
        return ParticleRepresentation(
            strategy=self.distribution_strategy,
            activity=self.activity_strategy,
            surface=self.surface_strategy,
            distribution=resolved_masses,
            density=self.density,
            concentration=number_concentration,
            charge=self.charge,
            volume=self.volume,
        )
