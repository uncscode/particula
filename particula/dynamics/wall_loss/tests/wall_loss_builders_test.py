"""Tests for wall loss builders."""

import pytest

from particula.dynamics.wall_loss.wall_loss_builders import (
    ChargedWallLossBuilder,
    RectangularWallLossBuilder,
    SphericalWallLossBuilder,
)
from particula.dynamics.wall_loss.wall_loss_strategies import (
    ChargedWallLossStrategy,
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


def test_rectangular_zero_diffusivity_raises():
    """Zero wall eddy diffusivity is invalid."""
    builder = RectangularWallLossBuilder()
    with pytest.raises(ValueError):
        builder.set_wall_eddy_diffusivity(0.0)


def test_charged_builder_spherical_success():
    """Charged builder constructs spherical charged strategy."""
    strategy = (
        ChargedWallLossBuilder()
        .set_wall_eddy_diffusivity(0.001)
        .set_chamber_geometry("spherical")
        .set_chamber_radius(0.5)
        .set_wall_potential(1.0)
        .build()
    )
    assert isinstance(strategy, ChargedWallLossStrategy)
    assert strategy.chamber_geometry == "spherical"


def test_charged_builder_missing_geometry_raises():
    """Geometry is required before build."""
    builder = ChargedWallLossBuilder().set_wall_eddy_diffusivity(0.001)
    with pytest.raises(ValueError):
        builder.build()


def test_charged_builder_rectangular_missing_dimensions_raises():
    """Rectangular geometry requires chamber dimensions."""
    builder = (
        ChargedWallLossBuilder()
        .set_wall_eddy_diffusivity(0.001)
        .set_chamber_geometry("rectangular")
    )
    with pytest.raises(ValueError):
        builder.build()


def test_charged_builder_invalid_geometry_string():
    """Invalid geometry strings should raise ValueError."""
    builder = ChargedWallLossBuilder().set_wall_eddy_diffusivity(0.001)
    with pytest.raises(ValueError):
        builder.set_chamber_geometry("cylindrical")


def test_charged_builder_wall_electric_field_tuple_length_invalid():
    """Electric field tuples must have length three for rectangular geometry."""
    builder = ChargedWallLossBuilder()
    with pytest.raises(ValueError):
        builder.set_wall_electric_field((1.0, 2.0))


def test_charged_builder_wall_electric_field_nonfinite_raises():
    """Non-finite electric field values are rejected."""
    builder = ChargedWallLossBuilder()
    with pytest.raises(ValueError):
        builder.set_wall_electric_field(float("nan"))


def test_charged_builder_set_parameters_sets_optional_fields():
    """set_parameters wires optional potential and field into built strategy."""
    strategy = (
        ChargedWallLossBuilder()
        .set_parameters(
            {
                "wall_eddy_diffusivity": 0.001,
                "chamber_geometry": "rectangular",
                "chamber_dimensions": (1.0, 0.5, 0.5),
                "wall_potential": 2.0,
                "wall_electric_field": (10.0, 0.0, 0.0),
            }
        )
        .build()
    )
    assert strategy.wall_potential == pytest.approx(2.0)
    assert strategy.wall_electric_field == (10.0, 0.0, 0.0)
    assert strategy.chamber_geometry == "rectangular"
