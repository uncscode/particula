"""Tests for wall loss strategy classes.

Covers the abstract :class:`WallLossStrategy` interface and the
concrete :class:`SphericalWallLossStrategy` implementation.

This test module is located under ``particula/dynamics/tests`` to avoid
package import edge cases when running the full test suite
(``pytest particula``).
"""

import unittest

import numpy as np
from particula.dynamics import (
    RectangularWallLossStrategy as ExportedRectangularWallLossStrategy,
)
from particula.dynamics import (
    get_rectangle_wall_loss_rate,
)
from particula.dynamics.tests.wall_loss_native_helpers import (
    _concentration,
    _particle,
    _total,
)
from particula.dynamics.wall_loss.wall_loss_strategies import (
    RectangularWallLossStrategy,
    SphericalWallLossStrategy,
    WallLossStrategy,
)


class TestWallLossStrategies(unittest.TestCase):
    """Test suite for wall loss strategies."""

    def setUp(self):
        """Set up particle representations and strategies for testing."""
        self.temperature = 298.15  # K
        self.pressure = 101325.0  # Pa
        self.time_step = 1.0  # s

        # Native distributions carry explicit, distinct representation metadata
        self.particle = _particle()
        self.particle_continuous = _particle("continuous_pdf")

        # Particle-resolved representation with finite parcel volume
        self.particle_resolved = _particle("particle_resolved", 1e-6)

        self.strategy_discrete = SphericalWallLossStrategy(
            wall_eddy_diffusivity=1e-3,
            chamber_radius=0.5,
            distribution_type="discrete",
        )
        self.strategy_continuous = SphericalWallLossStrategy(
            wall_eddy_diffusivity=1e-3,
            chamber_radius=0.5,
            distribution_type="continuous_pdf",
        )
        self.strategy_particle_resolved = SphericalWallLossStrategy(
            wall_eddy_diffusivity=1e-3,
            chamber_radius=0.5,
            distribution_type="particle_resolved",
        )

    def test_abc_cannot_be_instantiated(self):
        """WallLossStrategy ABC cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            WallLossStrategy(wall_eddy_diffusivity=1e-3)  # type: ignore

    def test_invalid_distribution_type_raises(self):
        """Invalid distribution type raises a ValueError."""
        with self.assertRaises(ValueError):
            SphericalWallLossStrategy(
                wall_eddy_diffusivity=1e-3,
                chamber_radius=0.5,
                distribution_type="invalid",
            )

    def test_initialization_stores_parameters(self):
        """Initialization stores wall eddy diffusivity and chamber radius."""
        strategy = SphericalWallLossStrategy(
            wall_eddy_diffusivity=2e-3,
            chamber_radius=1.0,
            distribution_type="discrete",
        )
        self.assertEqual(strategy.wall_eddy_diffusivity, 2e-3)
        self.assertEqual(strategy.chamber_radius, 1.0)
        self.assertEqual(strategy.distribution_type, "discrete")

    def test_loss_coefficient_is_positive(self):
        """Loss coefficient must be strictly positive for typical states."""
        coefficient = self.strategy_discrete.loss_coefficient(
            particle=self.particle,
            temperature=self.temperature,
            pressure=self.pressure,
        )
        self.assertTrue(np.all(coefficient > 0.0))

    def test_loss_rate_is_negative(self):
        """Loss rate must be non-positive (indicating loss)."""
        rate = self.strategy_discrete.rate(
            particle=self.particle,
            temperature=self.temperature,
            pressure=self.pressure,
        )
        self.assertTrue(np.all(rate <= 0.0))

    def test_step_reduces_concentration_discrete(self):
        """Step must reduce total concentration for discrete distributions."""
        initial_total = _total(self.particle)
        self.strategy_discrete.step(
            particle=self.particle,
            temperature=self.temperature,
            pressure=self.pressure,
            time_step=self.time_step,
        )
        final_total = _total(self.particle)
        self.assertLessEqual(final_total, initial_total)

    def test_step_reduces_concentration_continuous_pdf(self):
        """Step must reduce concentration for continuous_pdf distributions."""
        initial_total = _total(self.particle_continuous)
        self.strategy_continuous.step(
            particle=self.particle_continuous,
            temperature=self.temperature,
            pressure=self.pressure,
            time_step=self.time_step,
        )
        final_total = _total(self.particle_continuous)
        self.assertLessEqual(final_total, initial_total)

    def test_step_particle_resolved_reduces_concentration(self):
        """Particle-resolved step should reduce total concentration."""
        initial_total = _total(self.particle_resolved)
        self.strategy_particle_resolved.step(
            particle=self.particle_resolved,
            temperature=self.temperature,
            pressure=self.pressure,
            time_step=100.0,
        )
        final_total = _total(self.particle_resolved)
        self.assertLessEqual(final_total, initial_total)

    def test_zero_concentration_edge_case(self):
        """Zero concentration should remain zero after step."""
        zero_particle = _particle()
        zero_particle.concentration[...] = 0.0

        rate = self.strategy_discrete.rate(
            particle=zero_particle,
            temperature=self.temperature,
            pressure=self.pressure,
        )
        self.assertTrue(np.allclose(rate, 0.0))

        initial_total = _total(zero_particle)
        self.strategy_discrete.step(
            particle=zero_particle,
            temperature=self.temperature,
            pressure=self.pressure,
            time_step=self.time_step,
        )
        final_total = _total(zero_particle)
        self.assertEqual(final_total, initial_total)


class TestRectangularWallLossStrategies(unittest.TestCase):
    """Mirrored rectangular wall loss tests via dynamics imports."""

    def setUp(self):
        """Set up rectangular strategy fixtures."""
        self.temperature = 298.15
        self.pressure = 101325.0
        self.time_step = 1.0
        self.chamber_dimensions = (1.0, 0.6, 0.3)

        self.particle = _particle()
        self.particle_resolved = _particle("particle_resolved", 1e-6)

        self.strategy_discrete = RectangularWallLossStrategy(
            wall_eddy_diffusivity=1e-3,
            chamber_dimensions=self.chamber_dimensions,
            distribution_type="discrete",
        )
        self.strategy_particle_resolved = RectangularWallLossStrategy(
            wall_eddy_diffusivity=1e-3,
            chamber_dimensions=self.chamber_dimensions,
            distribution_type="particle_resolved",
        )

    def test_export_available_via_dynamics(self):
        """Rectangular strategy should be importable from particula.dynamics."""
        exported = ExportedRectangularWallLossStrategy(
            wall_eddy_diffusivity=1e-3,
            chamber_dimensions=self.chamber_dimensions,
            distribution_type="discrete",
        )
        self.assertIsInstance(exported, ExportedRectangularWallLossStrategy)
        self.assertEqual(
            exported.__class__.__name__, "RectangularWallLossStrategy"
        )

    def test_invalid_distribution_type_raises(self):
        """Invalid distribution type should raise ValueError."""
        with self.assertRaises(ValueError):
            RectangularWallLossStrategy(
                wall_eddy_diffusivity=1e-3,
                chamber_dimensions=self.chamber_dimensions,
                distribution_type="invalid",
            )

    def test_non_positive_wall_eddy_diffusivity_raises(self):
        """Non-positive wall eddy diffusivity should raise ValueError."""
        with self.assertRaises(ValueError):
            RectangularWallLossStrategy(
                wall_eddy_diffusivity=-1.0,
                chamber_dimensions=self.chamber_dimensions,
            )

    def test_loss_rate_is_negative_and_finite(self):
        """Loss rate should be finite and non-positive for valid inputs."""
        rate = self.strategy_discrete.rate(
            particle=self.particle,
            temperature=self.temperature,
            pressure=self.pressure,
        )
        self.assertTrue(np.all(np.isfinite(rate)))
        self.assertTrue(np.all(rate <= 0.0))

    def test_zero_concentration_smoke(self):
        """Zero concentration remains zero via dynamics import path."""
        zero_particle = _particle()
        zero_particle.concentration[...] = 0.0
        rate = self.strategy_discrete.rate(
            particle=zero_particle,
            temperature=self.temperature,
            pressure=self.pressure,
        )
        self.assertTrue(np.all(np.isfinite(rate)))
        self.assertTrue(np.allclose(rate, 0.0))
        initial_total = _total(zero_particle)
        self.strategy_discrete.step(
            particle=zero_particle,
            temperature=self.temperature,
            pressure=self.pressure,
            time_step=self.time_step,
        )
        final_total = _total(zero_particle)
        self.assertEqual(final_total, initial_total)

    def test_parity_with_rectangle_wall_loss_rate(self):
        """Strategy rate matches standalone rectangle wall loss rate."""
        strategy_rate = self.strategy_discrete.rate(
            particle=self.particle,
            temperature=self.temperature,
            pressure=self.pressure,
        )

        standalone_rate = get_rectangle_wall_loss_rate(
            wall_eddy_diffusivity=self.strategy_discrete.wall_eddy_diffusivity,
            particle_radius=self.particle.radii[0],
            particle_density=self.particle.effective_density[0],
            particle_concentration=_concentration(self.particle),
            temperature=self.temperature,
            pressure=self.pressure,
            chamber_dimensions=self.chamber_dimensions,
        )
        np.testing.assert_allclose(strategy_rate, standalone_rate, rtol=1e-10)

    def test_particle_resolved_shape(self):
        """Particle-resolved path returns matching shape and finiteness."""
        rate = self.strategy_particle_resolved.rate(
            particle=self.particle_resolved,
            temperature=self.temperature,
            pressure=self.pressure,
        )
        self.assertEqual(
            rate.shape, _concentration(self.particle_resolved).shape
        )
        self.assertTrue(np.all(np.isfinite(rate)))
