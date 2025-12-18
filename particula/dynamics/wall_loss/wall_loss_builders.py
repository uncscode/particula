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

    Constructs a spherical wall loss strategy with validated parameters and
    unit conversion support.
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
            Configured :class:`SphericalWallLossStrategy` instance.

        Raises:
            ValueError: If required parameters are not set.
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

    Constructs a rectangular wall loss strategy with validated parameters and
    unit conversion support.
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
            Configured :class:`RectangularWallLossStrategy` instance.

        Raises:
            ValueError: If required parameters are not set.
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
