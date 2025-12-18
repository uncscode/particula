"""Tests for wall loss builders."""

import pytest

from particula.dynamics.wall_loss.wall_loss_builders import (
    RectangularWallLossBuilder,
    SphericalWallLossBuilder,
)
from particula.dynamics.wall_loss.wall_loss_strategies import (
    RectangularWallLossStrategy,
    SphericalWallLossStrategy,
)


def test_spherical_builder_success():
    """Spherical builder constructs strategy with required parameters."""
    strategy = (
        SphericalWallLossBuilder()
        .set_wall_eddy_diffusivity(0.001)
        .set_chamber_radius(0.5)
        .build()
    )
    assert isinstance(strategy, SphericalWallLossStrategy)
    assert strategy.wall_eddy_diffusivity == pytest.approx(0.001)
    assert strategy.chamber_radius == pytest.approx(0.5)
    assert strategy.distribution_type == "discrete"


def test_spherical_builder_unit_conversion():
    """Radius converts from centimeters to meters."""
    strategy = (
        SphericalWallLossBuilder()
        .set_wall_eddy_diffusivity(0.001)
        .set_chamber_radius(50.0, "cm")
        .build()
    )
    assert strategy.chamber_radius == pytest.approx(0.5)


def test_spherical_missing_parameter_raises():
    """Missing chamber radius triggers pre-build validation error."""
    builder = SphericalWallLossBuilder().set_wall_eddy_diffusivity(0.001)
    with pytest.raises(ValueError, match="Required parameter"):
        builder.build()


def test_spherical_invalid_distribution_type_raises():
    """Invalid distribution type is rejected."""
    builder = SphericalWallLossBuilder()
    with pytest.raises(ValueError, match="distribution_type must be one of"):
        builder.set_distribution_type("invalid")


def test_spherical_negative_diffusivity_raises():
    """Negative wall eddy diffusivity is invalid."""
    builder = SphericalWallLossBuilder()
    with pytest.raises(ValueError):
        builder.set_wall_eddy_diffusivity(-0.1)


def test_spherical_zero_diffusivity_raises():
    """Zero wall eddy diffusivity is invalid."""
    builder = SphericalWallLossBuilder()
    with pytest.raises(ValueError):
        builder.set_wall_eddy_diffusivity(0.0)


def test_spherical_zero_radius_raises():
    """Non-positive chamber radius is rejected."""
    builder = SphericalWallLossBuilder().set_wall_eddy_diffusivity(0.001)
    with pytest.raises(ValueError):
        builder.set_chamber_radius(0.0)


def test_spherical_method_chaining_returns_self():
    """Builder methods support chaining semantics."""
    builder = SphericalWallLossBuilder()
    result = builder.set_wall_eddy_diffusivity(0.001)
    assert result is builder


def test_rectangular_builder_success():
    """Rectangular builder constructs strategy with required parameters."""
    strategy = (
        RectangularWallLossBuilder()
        .set_wall_eddy_diffusivity(0.0001)
        .set_chamber_dimensions((1.0, 0.5, 0.5))
        .build()
    )
    assert isinstance(strategy, RectangularWallLossStrategy)
    assert strategy.wall_eddy_diffusivity == pytest.approx(0.0001)
    assert strategy.chamber_dimensions == pytest.approx((1.0, 0.5, 0.5))
    assert strategy.distribution_type == "discrete"


def test_rectangular_unit_conversion():
    """Dimensions convert from centimeters to meters."""
    strategy = (
        RectangularWallLossBuilder()
        .set_wall_eddy_diffusivity(0.0001)
        .set_chamber_dimensions((100.0, 50.0, 50.0), "cm")
        .build()
    )
    assert strategy.chamber_dimensions == pytest.approx((1.0, 0.5, 0.5))


def test_rectangular_invalid_distribution_type_raises():
    """Invalid distribution type raises ValueError."""
    builder = RectangularWallLossBuilder()
    with pytest.raises(ValueError, match="distribution_type must be one of"):
        builder.set_distribution_type("invalid")


def test_rectangular_missing_parameter_raises():
    """Missing chamber dimensions triggers error."""
    builder = RectangularWallLossBuilder().set_wall_eddy_diffusivity(0.0001)
    with pytest.raises(ValueError, match="Required parameter"):
        builder.build()


def test_rectangular_invalid_dimensions_length():
    """Chamber dimensions must have exactly three values."""
    builder = RectangularWallLossBuilder().set_wall_eddy_diffusivity(0.0001)
    with pytest.raises(ValueError, match="three values"):
        builder.set_chamber_dimensions((1.0, 0.5))


def test_rectangular_non_positive_dimensions_raise():
    """Non-positive chamber dimensions are rejected."""
    builder = RectangularWallLossBuilder().set_wall_eddy_diffusivity(0.0001)
    with pytest.raises(ValueError):
        builder.set_chamber_dimensions((1.0, 0.0, 0.5))


def test_rectangular_negative_diffusivity_raises():
    """Negative wall eddy diffusivity is invalid."""
    builder = RectangularWallLossBuilder()
    with pytest.raises(ValueError):
        builder.set_wall_eddy_diffusivity(-0.1)
