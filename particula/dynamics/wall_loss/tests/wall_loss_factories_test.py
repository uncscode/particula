"""Tests for wall loss factory."""

import pytest

from particula.dynamics.wall_loss.wall_loss_factories import WallLossFactory
from particula.dynamics.wall_loss.wall_loss_strategies import (
    ChargedWallLossStrategy,
    RectangularWallLossStrategy,
    SphericalWallLossStrategy,
)


def test_factory_creates_spherical_strategy():
    """Factory builds a spherical wall loss strategy."""
    strategy = WallLossFactory().get_strategy(
        strategy_type="spherical",
        parameters={
            "wall_eddy_diffusivity": 0.001,
            "chamber_radius": 0.5,
        },
    )
    assert isinstance(strategy, SphericalWallLossStrategy)
    assert strategy.wall_eddy_diffusivity == pytest.approx(0.001)
    assert strategy.chamber_radius == pytest.approx(0.5)


def test_factory_creates_rectangular_strategy():
    """Factory builds a rectangular wall loss strategy."""
    strategy = WallLossFactory().get_strategy(
        strategy_type="rectangular",
        parameters={
            "wall_eddy_diffusivity": 0.0001,
            "chamber_dimensions": (1.0, 0.5, 0.5),
            "distribution_type": "continuous_pdf",
        },
    )
    assert isinstance(strategy, RectangularWallLossStrategy)
    assert strategy.wall_eddy_diffusivity == pytest.approx(0.0001)
    assert strategy.chamber_dimensions == pytest.approx((1.0, 0.5, 0.5))
    assert strategy.distribution_type == "continuous_pdf"


def test_factory_invalid_strategy_type_raises():
    """Unknown strategy type raises ValueError."""
    with pytest.raises(ValueError, match="Unknown strategy type"):
        WallLossFactory().get_strategy(
            strategy_type="cylindrical",
            parameters={},
        )


def test_factory_get_builders_contains_expected_keys():
    """Factory exposes both spherical and rectangular builders."""
    builders = WallLossFactory().get_builders()
    assert set(builders) == {"spherical", "rectangular", "charged"}


def test_factory_creates_charged_strategy():
    """Factory builds a charged wall loss strategy."""
    strategy = WallLossFactory().get_strategy(
        strategy_type="charged",
        parameters={
            "wall_eddy_diffusivity": 0.001,
            "chamber_geometry": "spherical",
            "chamber_radius": 0.5,
            "wall_potential": 0.0,
        },
    )
    assert isinstance(strategy, ChargedWallLossStrategy)
