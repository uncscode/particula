"""Builders for legacy particle representations.

This module provides builder classes and normalization helpers for creating
deprecated :class:`ParticleRepresentation` facades with mass-based,
radius-based, and particle-resolved distributions.
"""

import logging
from typing import Optional, Self, cast

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


def _normalize_resolved_density_input(
    density: float | NDArray[np.float64],
    mass: NDArray[np.float64],
) -> float | NDArray[np.float64]:
    """Normalize resolved density input to the expected species shape.

    Args:
        density: Density specified as a scalar or array-like value.
        mass: Resolved mass array used as the target shape reference.

    Returns:
        A scalar density or 1D density array aligned with the species axis
        expected by :class:`ParticleRepresentation`.
    """
    density_array = np.asarray(density, dtype=np.float64)
    if density_array.ndim == 2 and density_array.shape == mass.shape:
        if mass.shape[0] == 0:
            return np.zeros(mass.shape[-1], dtype=np.float64)
        return density_array[0, :]
    if density_array.ndim == 2 and 1 in density_array.shape:
        return np.ravel(density_array)
    return density


def _normalize_resolved_charge_input(
    charge: float | NDArray[np.float64] | None,
    mass: NDArray[np.float64],
) -> float | NDArray[np.float64] | None:
    """Normalize resolved charge input to the expected particle shape.

    Args:
        charge: Charge specified as ``None``, a scalar, or an array-like value.
        mass: Resolved mass array used as the target shape reference.

    Returns:
        ``None``, a scalar charge, or a 1D charge array aligned with the
        particle axis expected by :class:`ParticleRepresentation`.
    """
    if charge is None:
        return None

    charge_array = np.asarray(charge, dtype=np.float64)
    if charge_array.ndim == 2 and charge_array.shape == mass.shape:
        if charge_array.shape[0] == 1:
            return np.ravel(charge_array)
        return charge_array[:, 0]
    if charge_array.ndim == 2 and 1 in charge_array.shape:
        return np.ravel(charge_array)
    return charge


# pylint: disable=too-few-public-methods
class BuilderSurfaceStrategyMixin:
    """Mixin that stores a surface strategy for representation builders."""

    surface_strategy: SurfaceStrategy | None

    def __init__(self) -> None:
        """Initialize the mixin with no configured surface strategy."""
        self.surface_strategy = None

    def set_surface_strategy(
        self,
        surface_strategy: SurfaceStrategy,
        surface_strategy_units: Optional[str] = None,
    ) -> Self:
        """Assign the surface strategy for the particle representation.

        Args:
            surface_strategy: Strategy controlling surface properties.
            surface_strategy_units: Unused units argument kept for builder
                interface compatibility.

        Returns:
            The builder instance for chaining.
        """
        if surface_strategy_units is not None:
            logger.warning("Ignoring units for surface strategy parameter.")
        self.surface_strategy = surface_strategy
        return self


# pylint: disable=too-few-public-methods
class BuilderDistributionStrategyMixin:
    """Mixin that stores a distribution strategy for builders."""

    distribution_strategy: DistributionStrategy | None

    def __init__(self) -> None:
        """Initialize the mixin with no configured distribution strategy."""
        self.distribution_strategy = None

    def set_distribution_strategy(
        self,
        distribution_strategy: DistributionStrategy,
        distribution_strategy_units: Optional[str] = None,
    ) -> Self:
        """Assign the distribution strategy for the particle representation.

        Args:
            distribution_strategy: Strategy defining how the particle
                distribution is interpreted.
            distribution_strategy_units: Unused units argument kept for builder
                interface compatibility.

        Returns:
            The builder instance for chaining.
        """
        if distribution_strategy_units is not None:
            logger.warning(
                "Ignoring units for distribution strategy parameter."
            )
        self.distribution_strategy = distribution_strategy
        return self


# pylint: disable=too-few-public-methods
class BuilderActivityStrategyMixin:
    """Mixin that stores an activity strategy for builders."""

    activity_strategy: ActivityStrategy | None

    def __init__(self) -> None:
        """Initialize the mixin with no configured activity strategy."""
        self.activity_strategy = None

    def set_activity_strategy(
        self,
        activity_strategy: ActivityStrategy,
        activity_strategy_units: Optional[str] = None,
    ) -> Self:
        """Assign the activity strategy for the particle representation.

        Args:
            activity_strategy: Strategy defining particle-phase activities.
            activity_strategy_units: Unused units argument kept for builder
                interface compatibility.

        Returns:
            The builder instance for chaining.
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
    """Build particle representations from mass-based distributions."""

    def __init__(self) -> None:
        """Initialize required fields for mass-based representations."""
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
        """Build a mass-based particle representation.

        Returns:
            A particle representation configured with the stored distribution,
            activity, surface, density, concentration, and charge data.
        """
        self.pre_build_check()
        distribution_strategy = cast(
            DistributionStrategy, self.distribution_strategy
        )
        activity_strategy = cast(ActivityStrategy, self.activity_strategy)
        surface_strategy = cast(SurfaceStrategy, self.surface_strategy)
        return ParticleRepresentation(
            strategy=distribution_strategy,
            activity=activity_strategy,
            surface=surface_strategy,
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
    """Build particle representations from radius-based distributions."""

    def __init__(self) -> None:
        """Initialize required fields for radius-based representations."""
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
        """Build a radius-based particle representation.

        Returns:
            A particle representation configured with the stored radius,
            strategy, activity, surface, density, concentration, and charge
            data.
        """
        self.pre_build_check()
        distribution_strategy = cast(
            DistributionStrategy, self.distribution_strategy
        )
        activity_strategy = cast(ActivityStrategy, self.activity_strategy)
        surface_strategy = cast(SurfaceStrategy, self.surface_strategy)
        return ParticleRepresentation(
            strategy=distribution_strategy,
            activity=activity_strategy,
            surface=surface_strategy,
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
    """Build preset radius-bin representations from lognormal parameters."""

    def __init__(self) -> None:
        """Initialize defaults for preset radius-bin representations."""
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
    ) -> Self:
        """Assign the radius-bin grid used to generate the distribution.

        Args:
            radius_bins: Radius bin values.
            radius_bins_units: Units for ``radius_bins``. Defaults to meters.

        Returns:
            The builder instance for chaining.

        Raises:
            ValueError: If any radius bin is negative.
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
    ) -> Self:
        """Choose whether the preset distribution is generated as a PDF or PMF.

        Args:
            distribution_type: Either ``"pdf"`` or ``"pmf"``.
            distribution_type_units: Unused units argument kept for builder
                interface compatibility.

        Returns:
            The builder instance for chaining.

        Raises:
            ValueError: If ``distribution_type`` is not supported.
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
        """Build a radius-bin representation from preset lognormal inputs.

        Returns:
            A particle representation containing the generated radius-bin
            concentration distribution together with the configured strategies.

        Raises:
            ValueError: If the configured distribution type is unsupported.
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
        distribution_strategy = cast(
            DistributionStrategy, self.distribution_strategy
        )
        activity_strategy = cast(ActivityStrategy, self.activity_strategy)
        surface_strategy = cast(SurfaceStrategy, self.surface_strategy)
        return ParticleRepresentation(
            strategy=distribution_strategy,
            activity=activity_strategy,
            surface=surface_strategy,
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
    """Build particle-resolved representations from resolved mass arrays."""

    def __init__(self) -> None:
        """Initialize required fields for resolved-mass representations."""
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
        """Build a particle-resolved representation from resolved masses.

        Returns:
            A particle representation configured with per-particle masses,
            normalized density and charge inputs, and the stored strategies.
        """
        number_concentration = np.ones(self.mass.shape[0], dtype=np.float64)

        density = _normalize_resolved_density_input(self.density, self.mass)
        charge = _normalize_resolved_charge_input(self.charge, self.mass)

        self.pre_build_check()
        distribution_strategy = cast(
            DistributionStrategy, self.distribution_strategy
        )
        activity_strategy = cast(ActivityStrategy, self.activity_strategy)
        surface_strategy = cast(SurfaceStrategy, self.surface_strategy)
        return ParticleRepresentation(
            strategy=distribution_strategy,
            activity=activity_strategy,
            surface=surface_strategy,
            distribution=self.mass,
            density=density,
            concentration=number_concentration,
            charge=charge,
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
    """Build preset particle-resolved representations from lognormal inputs."""

    def __init__(self) -> None:
        """Initialize defaults for preset particle-resolved representations."""
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
        """Build a preset particle-resolved representation.

        Returns:
            A particle representation whose per-particle masses are sampled
            from the configured lognormal distribution.
        """
        resolved_radii = get_lognormal_sample_distribution(
            mode=self.mode,
            geometric_standard_deviation=self.geometric_standard_deviation,
            number_of_particles=self.number_concentration,
            number_of_samples=self.particle_resolved_count,
        )
        # convert radii to masses
        resolved_masses = np.asarray(
            4 / 3 * np.pi * resolved_radii**3 * self.density,
            dtype=np.float64,
        )
        number_concentration = np.ones(
            resolved_masses.shape[0],
            dtype=np.float64,
        )

        density = _normalize_resolved_density_input(
            self.density, resolved_masses
        )
        charge = _normalize_resolved_charge_input(self.charge, resolved_masses)
        if (
            isinstance(charge, np.ndarray)
            and charge.ndim == 1
            and charge.size != number_concentration.size
            and np.all(charge == 0)
        ):
            charge = np.zeros(number_concentration.size, dtype=np.float64)

        self.pre_build_check()
        distribution_strategy = cast(
            DistributionStrategy, self.distribution_strategy
        )
        activity_strategy = cast(ActivityStrategy, self.activity_strategy)
        surface_strategy = cast(SurfaceStrategy, self.surface_strategy)
        return ParticleRepresentation(
            strategy=distribution_strategy,
            activity=activity_strategy,
            surface=surface_strategy,
            distribution=resolved_masses,
            density=density,
            concentration=number_concentration,
            charge=charge,
            volume=self.volume,
        )
