"""Wall loss strategy builders.

Provides builder classes for constructing wall loss strategies with validated
parameters and unit conversion support.
"""

from particula.abc_builder import BuilderABC
from particula.builder_mixin import (
    BuilderChamberDimensionsMixin,
    BuilderChamberRadiusMixin,
    BuilderDistributionTypeMixin,
    BuilderWallEddyDiffusivityMixin,
)
from particula.dynamics.wall_loss.wall_loss_strategies import (
    RectangularWallLossStrategy,
    SphericalWallLossStrategy,
)


class SphericalWallLossBuilder(
    BuilderABC,
    BuilderWallEddyDiffusivityMixin,
    BuilderChamberRadiusMixin,
    BuilderDistributionTypeMixin,
):
    """Builder for spherical wall loss strategies.

    Validates wall eddy diffusivity, chamber radius, and distribution type
    before constructing :class:`SphericalWallLossStrategy`.

    Attributes:
        wall_eddy_diffusivity: Wall eddy diffusivity in 1/s.
        chamber_radius: Chamber radius in meters.
        distribution_type: Distribution type for the strategy output.
    """

    def __init__(self):
        """Initialize the spherical wall loss builder."""
        BuilderABC.__init__(
            self,
            required_parameters=["wall_eddy_diffusivity", "chamber_radius"],
        )
        BuilderWallEddyDiffusivityMixin.__init__(self)
        BuilderChamberRadiusMixin.__init__(self)
        BuilderDistributionTypeMixin.__init__(self)

    def build(self) -> SphericalWallLossStrategy:
        """Build and return the spherical wall loss strategy.

        Returns:
            SphericalWallLossStrategy: Configured wall loss strategy.

        Raises:
            ValueError: If required parameters are missing or invalid.
        """
        self.pre_build_check()
        wall_eddy_diffusivity = self.wall_eddy_diffusivity
        chamber_radius = self.chamber_radius
        if wall_eddy_diffusivity is None or chamber_radius is None:
            msg = "Required parameters not set."
            raise ValueError(msg)
        return SphericalWallLossStrategy(
            wall_eddy_diffusivity=wall_eddy_diffusivity,
            chamber_radius=chamber_radius,
            distribution_type=self.distribution_type,
        )


class RectangularWallLossBuilder(
    BuilderABC,
    BuilderWallEddyDiffusivityMixin,
    BuilderChamberDimensionsMixin,
    BuilderDistributionTypeMixin,
):
    """Builder for rectangular wall loss strategies.

    Validates wall eddy diffusivity, chamber dimensions, and distribution type
    before constructing :class:`RectangularWallLossStrategy`.

    Attributes:
        wall_eddy_diffusivity: Wall eddy diffusivity in 1/s.
        chamber_dimensions: Tuple of (length, width, height) in meters.
        distribution_type: Distribution type for the strategy output.
    """

    def __init__(self):
        """Initialize the rectangular wall loss builder."""
        BuilderABC.__init__(
            self,
            required_parameters=[
                "wall_eddy_diffusivity",
                "chamber_dimensions",
            ],
        )
        BuilderWallEddyDiffusivityMixin.__init__(self)
        BuilderChamberDimensionsMixin.__init__(self)
        BuilderDistributionTypeMixin.__init__(self)

    def build(self) -> RectangularWallLossStrategy:
        """Build and return the rectangular wall loss strategy.

        Returns:
            RectangularWallLossStrategy: Configured wall loss strategy.

        Raises:
            ValueError: If required parameters are missing or invalid.
        """
        self.pre_build_check()
        wall_eddy_diffusivity = self.wall_eddy_diffusivity
        chamber_dimensions = self.chamber_dimensions
        if wall_eddy_diffusivity is None or chamber_dimensions is None:
            msg = "Required parameters not set."
            raise ValueError(msg)
        return RectangularWallLossStrategy(
            wall_eddy_diffusivity=wall_eddy_diffusivity,
            chamber_dimensions=chamber_dimensions,
            distribution_type=self.distribution_type,
        )
