"""Tests for wall loss strategy classes.

Covers the abstract :class:`WallLossStrategy` interface and the
concrete :class:`SphericalWallLossStrategy` implementation.

This test module is located under ``particula/dynamics/tests`` to avoid
package import edge cases when running the full test suite
(``pytest particula``).
"""

import importlib.util
import pathlib
import sys
import unittest

import numpy as np

from particula.particles import (
    PresetParticleRadiusBuilder,
    PresetResolvedParticleMassBuilder,
)

# Import the wall loss strategies directly from the source file to avoid
# environment-specific package resolution issues for
# ``particula.dynamics.wall_loss`` when tests are run via ADW tooling.
_MODULE_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "wall_loss"
    / "wall_loss_strategies.py"
)

_spec = importlib.util.spec_from_file_location(
    "_wall_loss_strategies_test_module",
    _MODULE_PATH,
)
assert _spec and _spec.loader is not None
_wall_loss_strategies = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _wall_loss_strategies
_spec.loader.exec_module(_wall_loss_strategies)

SphericalWallLossStrategy = _wall_loss_strategies.SphericalWallLossStrategy
WallLossStrategy = _wall_loss_strategies.WallLossStrategy


class TestWallLossStrategies(unittest.TestCase):
    """Test suite for wall loss strategies."""

    def setUp(self):
        """Set up particle representations and strategies for testing."""
        self.temperature = 298.15  # K
        self.pressure = 101325.0  # Pa
        self.time_step = 1.0  # s

        # Discrete/continuous representations share the same preset builder
        self.particle = PresetParticleRadiusBuilder().build()

        # Particle-resolved representation with finite parcel volume
        self.particle_resolved = (
            PresetResolvedParticleMassBuilder().set_volume(1e-6, "m^3").build()
        )

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
        initial_total = self.particle.get_total_concentration()
        self.strategy_discrete.step(
            particle=self.particle,
            temperature=self.temperature,
            pressure=self.pressure,
            time_step=self.time_step,
        )
        final_total = self.particle.get_total_concentration()
        self.assertLessEqual(final_total, initial_total)

    def test_step_reduces_concentration_continuous_pdf(self):
        """Step must reduce concentration for continuous_pdf distributions."""
        initial_total = self.particle.get_total_concentration()
        self.strategy_continuous.step(
            particle=self.particle,
            temperature=self.temperature,
            pressure=self.pressure,
            time_step=self.time_step,
        )
        final_total = self.particle.get_total_concentration()
        self.assertLessEqual(final_total, initial_total)

    def test_step_particle_resolved_reduces_concentration(self):
        """Particle-resolved step should reduce total concentration."""
        initial_total = self.particle_resolved.get_total_concentration()
        self.strategy_particle_resolved.step(
            particle=self.particle_resolved,
            temperature=self.temperature,
            pressure=self.pressure,
            time_step=100.0,
        )
        final_total = self.particle_resolved.get_total_concentration()
        self.assertLessEqual(final_total, initial_total)

    def test_zero_concentration_edge_case(self):
        """Zero concentration should remain zero after step."""
        zero_particle = PresetParticleRadiusBuilder().build()
        zero_particle.concentration[...] = 0.0

        rate = self.strategy_discrete.rate(
            particle=zero_particle,
            temperature=self.temperature,
            pressure=self.pressure,
        )
        self.assertTrue(np.allclose(rate, 0.0))

        initial_total = zero_particle.get_total_concentration()
        self.strategy_discrete.step(
            particle=zero_particle,
            temperature=self.temperature,
            pressure=self.pressure,
            time_step=self.time_step,
        )
        final_total = zero_particle.get_total_concentration()
        self.assertEqual(final_total, initial_total)
