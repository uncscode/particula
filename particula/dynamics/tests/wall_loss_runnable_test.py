"""Tests for the WallLoss runnable process."""

import unittest
import numpy as np

from particula.particles import PresetParticleRadiusBuilder
from particula.gas import (
    AtmosphereBuilder,
    GasSpeciesBuilder,
    ConstantVaporPressureStrategy,
)
from particula.aerosol import Aerosol
from particula.dynamics.wall_loss_strategy import WallLossStrategy
from particula.dynamics.particle_process import WallLoss


class TestWallLossRunnable(unittest.TestCase):
    """Verify wall loss process modifies aerosol particles."""

    def setUp(self):
        particle = PresetParticleRadiusBuilder().build()
        gas = (
            GasSpeciesBuilder()
            .set_name("air")
            .set_molar_mass(0.029, "kg/mol")
            .set_vapor_pressure_strategy(ConstantVaporPressureStrategy(0.0))
            .set_partitioning(True)
            .set_concentration(1.0, "kg/m^3")
            .build()
        )
        atmosphere = (
            AtmosphereBuilder()
            .set_temperature(298.15, "K")
            .set_pressure(101325.0, "Pa")
            .set_more_partitioning_species(gas)
            .build()
        )
        self.aerosol = Aerosol(atmosphere=atmosphere, particles=particle)
        self.strategy = WallLossStrategy(
            wall_eddy_diffusivity=0.1,
            chamber_radius=1.0,
        )
        self.process = WallLoss(self.strategy)

    def test_execute_updates_particles(self):
        """Execute should apply concentration change."""
        rate = self.strategy.rate(
            particle=self.aerosol.particles,
            temperature=self.aerosol.atmosphere.temperature,
            pressure=self.aerosol.atmosphere.total_pressure,
        )
        initial = self.aerosol.particles.get_concentration().copy()
        updated = self.process.execute(self.aerosol, time_step=2.0)
        expected = initial + rate * 2.0
        np.testing.assert_allclose(updated.particles.get_concentration(), expected)


if __name__ == "__main__":
    unittest.main()
