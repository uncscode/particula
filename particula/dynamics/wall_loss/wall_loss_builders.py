"""Wall loss strategy builders.

Provides builder classes for constructing wall loss strategies with validated
parameters and unit conversion support.
"""

from typing import Optional

from particula.abc_builder import BuilderABC
from particula.builder_mixin import (
    BuilderChamberDimensionsMixin,
    BuilderChamberRadiusMixin,
    BuilderDistributionTypeMixin,
    BuilderWallEddyDiffusivityMixin,
    BuilderWallElectricFieldMixin,
    BuilderWallPotentialMixin,
)
from particula.dynamics.wall_loss.wall_loss_strategies import (
    ChargedWallLossStrategy,
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


class ChargedWallLossBuilder(
    BuilderABC,
    BuilderWallEddyDiffusivityMixin,
    BuilderDistributionTypeMixin,
    BuilderChamberRadiusMixin,
    BuilderChamberDimensionsMixin,
    BuilderWallPotentialMixin,
    BuilderWallElectricFieldMixin,
):
    """Builder for charged wall loss strategies.

    Configures image-charge-enhanced wall loss with optional
    electric-field drift for spherical or rectangular chambers.
    Requires geometry plus the matching size parameter. Zero wall potential
    still allows charge-driven enhancement.
    """

    def __init__(self):
        """Initialize the charged wall loss builder."""
        BuilderABC.__init__(
            self,
            required_parameters=["wall_eddy_diffusivity", "chamber_geometry"],
        )
        BuilderWallEddyDiffusivityMixin.__init__(self)
        BuilderDistributionTypeMixin.__init__(self)
        BuilderChamberRadiusMixin.__init__(self)
        BuilderChamberDimensionsMixin.__init__(self)
        BuilderWallPotentialMixin.__init__(self)
        BuilderWallElectricFieldMixin.__init__(self)
        self.chamber_geometry: Optional[str] = None

    def set_chamber_geometry(self, chamber_geometry: str):
        """Set chamber geometry.

        Args:
            chamber_geometry: ``"spherical"`` or ``"rectangular"``.

        Returns:
            ChargedWallLossBuilder: Self for method chaining.

        Raises:
            ValueError: If ``chamber_geometry`` is not supported.
        """
        normalized = chamber_geometry.lower()
        if normalized not in {"spherical", "rectangular"}:
            raise ValueError(
                "chamber_geometry must be 'spherical' or 'rectangular'."
            )
        self.chamber_geometry = normalized
        return self

    def set_parameters(self, parameters: dict):
        """Set required and optional parameters from mapping.

        Args:
            parameters: Mapping containing required ``wall_eddy_diffusivity``
                and ``chamber_geometry`` plus optional geometry size,
                ``wall_potential``, and ``wall_electric_field`` entries with
                optional ``*_units`` keys.

        Returns:
            ChargedWallLossBuilder: Self for method chaining.

        Raises:
            ValueError: If required keys are missing or unexpected keys are
                provided.
        """
        required = {"wall_eddy_diffusivity", "chamber_geometry"}
        optional = {
            "chamber_radius",
            "chamber_dimensions",
            "wall_potential",
            "wall_electric_field",
        }
        valid = (
            required
            | optional
            | {f"{key}_units" for key in optional | required}
        )
        if missing := [key for key in required if key not in parameters]:
            raise ValueError(
                f"Missing required parameter(s): {', '.join(sorted(missing))}"
            )
        if invalid := [key for key in parameters if key not in valid]:
            raise ValueError(
                f"Trying to set an invalid parameter(s) '{invalid}'."
            )
        self.set_wall_eddy_diffusivity(
            parameters["wall_eddy_diffusivity"],
            parameters.get("wall_eddy_diffusivity_units", "1/s"),
        )
        self.set_chamber_geometry(parameters["chamber_geometry"])
        if "chamber_radius" in parameters:
            self.set_chamber_radius(
                parameters["chamber_radius"],
                parameters.get("chamber_radius_units", "m"),
            )
        if "chamber_dimensions" in parameters:
            self.set_chamber_dimensions(
                parameters["chamber_dimensions"],
                parameters.get("chamber_dimensions_units", "m"),
            )
        if "wall_potential" in parameters:
            self.set_wall_potential(
                parameters["wall_potential"],
                parameters.get("wall_potential_units", "V"),
            )
        if "wall_electric_field" in parameters:
            self.set_wall_electric_field(
                parameters["wall_electric_field"],
                parameters.get("wall_electric_field_units", "V/m"),
            )
        return self

    def build(self) -> ChargedWallLossStrategy:
        """Build and return a charged wall loss strategy.

        Returns:
            ChargedWallLossStrategy: Charged strategy with geometry-specific
                sizing and electrostatic settings.

        Raises:
            ValueError: If geometry is unset, matching size parameters are
                missing, or required values are not provided.
        """
        self.pre_build_check()
        if self.wall_eddy_diffusivity is None:
            raise ValueError("wall_eddy_diffusivity must be set before build")
        wall_eddy_diffusivity = float(self.wall_eddy_diffusivity)
        if self.chamber_geometry == "spherical":
            if self.chamber_radius is None:
                raise ValueError("chamber_radius must be set for spherical")
            return ChargedWallLossStrategy(
                wall_eddy_diffusivity=wall_eddy_diffusivity,
                chamber_geometry=self.chamber_geometry,
                chamber_radius=self.chamber_radius,
                wall_potential=self.wall_potential,
                wall_electric_field=self.wall_electric_field,
                distribution_type=self.distribution_type,
            )
        if self.chamber_geometry == "rectangular":
            if self.chamber_dimensions is None:
                raise ValueError(
                    "chamber_dimensions must be set for rectangular geometry"
                )
            return ChargedWallLossStrategy(
                wall_eddy_diffusivity=wall_eddy_diffusivity,
                chamber_geometry=self.chamber_geometry,
                chamber_dimensions=self.chamber_dimensions,
                wall_potential=self.wall_potential,
                wall_electric_field=self.wall_electric_field,
                distribution_type=self.distribution_type,
            )
        raise ValueError("chamber_geometry must be set before build()")
