"""Smoke tests for wall loss builders and factory re-exports."""

from particula import dynamics


def test_builders_are_reexported():
    """Builders are exposed via particula.dynamics."""
    spherical_builder = dynamics.SphericalWallLossBuilder()
    rectangular_builder = dynamics.RectangularWallLossBuilder()

    assert isinstance(spherical_builder, dynamics.SphericalWallLossBuilder)
    assert isinstance(rectangular_builder, dynamics.RectangularWallLossBuilder)


def test_factory_is_reexported():
    """Factory is exposed via particula.dynamics."""
    factory = dynamics.WallLossFactory()
    assert isinstance(factory, dynamics.WallLossFactory)
