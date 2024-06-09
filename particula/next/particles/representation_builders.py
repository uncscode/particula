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

from particula.util.input_handling import convert_units

from particula.next.abc_builder import (
    BuilderABC,
    BuilderDistributionStrategyMixin,
    BuilderSurfaceStrategyMixin,
    BuilderActivityStrategyMixin,
    BuilderRadiusMixin,
    BuilderConcentrationMixin,
    BuilderMassMixin,
    BuilderDensityMixin,
    BuilderChargeMixin,
)
from particula.next.particles.distribution_strategies import (
    RadiiBasedMovingBin,
)
from particula.next.particles.activity_strategies import (
    IdealActivityMass,
)
from particula.next.particles.surface_strategies import (
    SurfaceStrategyVolume,
)
from particula.next.particles.representation import ParticleRepresentation
from particula.next.particles.properties.lognormal_size_distribution import (
    lognormal_pdf_distribution, lognormal_pmf_distribution
)

logger = logging.getLogger("particula")


class MassParticleRepresentationBuilder(
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

    Methods:
        set_distribution_strategy(strategy): Set the DistributionStrategy.
        set_activity_strategy(strategy): Set the ActivityStrategy.
        set_surface_strategy(strategy): Set the SurfaceStrategy.
        set_mass(mass, mass_units): Set the mass of the particles. Default
            units are 'kg'.
        set_density(density, density_units): Set the density of the particles.
            Default units are 'kg/m**3'.
        set_concentration(concentration, concentration_units): Set the
            concentration of the particles. Default units are '/m**3'.
        set_charge(charge, charge_units): Set the number of charges.
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


class RadiusParticleRepresentationBuilder(
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

    Methods:
        set_distribution_strategy(strategy): Set the DistributionStrategy.
        set_activity_strategy(strategy): Set the ActivityStrategy.
        set_surface_strategy(strategy): Set the SurfaceStrategy.
        set_radius(radius, radius_units): Set the radius of the particles.
            Default units are 'm'.
        set_density(density, density_units): Set the density of the particles.
            Default units are 'kg/m**3'.
        set_concentration(concentration, concentration_units): Set the
            concentration of the particles. Default units are '1/m^3'.
        set_charge(charge, charge_units): Set the number of charges.
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


class LimitedRadiusParticleBuilder(
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

    Methods:
        set_mode(mode,mode_units): Set the mode(s) of the distribution.
            Default is np.array([100e-9, 1e-6]) meters.
        set_geometric_standard_deviation(
            geometric_standard_deviation,geometric_standard_deviation_units):
                Set the geometric standard deviation(s) of the distribution.
                Default is np.array([1.2, 1.4]).
        set_number_concentration(
            number_concentration,number_concentration_units): Set the
                number concentration of the distribution. Default is
                np.array([1e4*1e6, 1e3*1e6]) particles/m**3.
        set_radius_bins(radius_bins,radius_bins_units): Set the radius bins
            of the distribution. Default is np.logspace(-9, -4, 250), meters.
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

        # set defaults
        self.mode = np.array([100e-9, 1e-6])
        self.geometric_standard_deviation = np.array([1.2, 1.4])
        self.number_concentration = np.array([1e4 * 1e6, 1e3 * 1e6])
        self.radius_bins = np.logspace(-9, -4, 250)
        self.set_distribution_strategy(RadiiBasedMovingBin())
        self.set_activity_strategy(IdealActivityMass())
        self.set_surface_strategy(
            SurfaceStrategyVolume(surface_tension=0.072, density=1000)
        )
        self.set_density(1000, "kg/m^3")
        self.set_charge(0)
        self.distribution_type = "pdf"

    def set_mode(
        self,
        mode: NDArray[np.float_],
        mode_units: str = "m",
    ):
        """Set the modes for distribution

        Args:
            modes: The modes for the radius.
            modes_units: The units for the modes.
        """
        if np.any(mode < 0):
            message = "The mode must be positive."
            logger.error(message)
            raise ValueError(message)
        self.mode = mode * convert_units(mode_units, "m")
        return self

    def set_geometric_standard_deviation(
        self,
        geometric_standard_deviation: NDArray[np.float_],
        geometric_standard_deviation_units: Optional[str] = None,
    ):
        """Set the geometric standard deviation for the distribution

        Args:
            geometric_standard_deviation: The geometric standard deviation for
            the radius.
        """
        if np.any(geometric_standard_deviation < 0):
            message = "The geometric standard deviation must be positive."
            logger.error(message)
            raise ValueError(message)
        if geometric_standard_deviation_units is not None:
            logger.warning("Ignoring units for surface strategy parameter.")
        self.geometric_standard_deviation = geometric_standard_deviation
        return self

    def set_number_concentration(
        self,
        number_concentration: NDArray[np.float_],
        number_concentration_units: str = "1/m^3",
    ):
        """Set the number concentration for the distribution

        Args:
            number_concentration: The number concentration for the radius.
        """
        if np.any(number_concentration < 0):
            message = "The number concentration must be positive."
            logger.error(message)
            raise ValueError(message)
        self.number_concentration = number_concentration * convert_units(
            number_concentration_units, "1/m^3"
        )
        return self

    def set_radius_bins(
        self,
        radius_bins: NDArray[np.float_],
        radius_bins_units: str = "m",
    ):
        """Set the radius bins for the distribution

        Args:
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

        Args:
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
