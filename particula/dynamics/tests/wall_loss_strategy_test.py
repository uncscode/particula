"""Tests for WallLossStrategy."""

import unittest
import numpy as np

from particula.particles import PresetParticleRadiusBuilder
from particula.dynamics.wall_loss_strategy import WallLossStrategy
from particula.dynamics.wall_loss import get_spherical_wall_loss_rate


class TestWallLossStrategy(unittest.TestCase):
    """Test the wall loss strategy class."""

    def setUp(self):
        self.particle = PresetParticleRadiusBuilder().build()
        self.strategy = WallLossStrategy(
            wall_eddy_diffusivity=0.1,
            chamber_radius=1.0,
        )
        self.temperature = 298.15
        self.pressure = 101325.0

    def test_rate_matches_function(self):
        """Rate should match helper function."""
        expected = get_spherical_wall_loss_rate(
            wall_eddy_diffusivity=0.1,
            particle_radius=self.particle.get_radius(),
            particle_density=self.particle.get_density(),
            particle_concentration=self.particle.get_concentration(),
            temperature=self.temperature,
            pressure=self.pressure,
            chamber_radius=1.0,
        )
        result = self.strategy.rate(
            particle=self.particle,
            temperature=self.temperature,
            pressure=self.pressure,
        )
        np.testing.assert_allclose(result, expected)

    def test_step_updates_concentration(self):
        """Step should modify particle concentration."""
        rate = self.strategy.rate(
            particle=self.particle,
            temperature=self.temperature,
            pressure=self.pressure,
        )
        initial = self.particle.get_concentration().copy()
        self.strategy.step(
            particle=self.particle,
            temperature=self.temperature,
            pressure=self.pressure,
            time_step=10.0,
        )
        expected = initial + rate * 10.0
        np.testing.assert_allclose(self.particle.get_concentration(), expected)


if __name__ == "__main__":
    unittest.main()
