"""Tests for the Aerosol class.
"""

import unittest
import numpy as np
from particula.particles import PresetParticleRadiusBuilder
from particula.gas import (
    AtmosphereBuilder,
    VaporPressureFactory,
    GasSpeciesBuilder,
)
from particula.aerosol import Aerosol
from particula.gas.species import GasSpecies
from particula.particles.representation import ParticleRepresentation


class TestAerosol(unittest.TestCase):
    """Test class for the Aerosol class."""

    def setUp(self):
        """Set up the test case with an real Atmosphere
            and ParticleRepresentation.
        """
        # Glycerol gas
        molar_mass_glycerol = 92.09382e-3  # kg/mol
        parameters_clausius = {
            "latent_heat": 71.5*1e3 * molar_mass_glycerol,
            "latent_heat_units": "J/kg",
            "temperature_initial": 125.5+273.15,
            "temperature_initial_units": "K",
            "pressure_initial": 1e5,
            "pressure_initial_units": "Pa",
        }
        vapor_pressure_strategy = VaporPressureFactory().get_strategy(
            "clausius_clapeyron", parameters_clausius
        )

        sat_concentration = vapor_pressure_strategy.saturation_concentration(
            molar_mass_glycerol, 298.15
        )

        sat_factor = 0.5  # 50% of saturation concentration
        glycerol_gas = (
            GasSpeciesBuilder()
            .set_molar_mass(molar_mass_glycerol, "kg/mol")
            .set_vapor_pressure_strategy(vapor_pressure_strategy)
            .set_concentration(sat_concentration * sat_factor, "kg/m^3")
            .set_name("Glycerol")
            .set_condensable(True)
            .build()
        )

        self.atmosphere = (
            AtmosphereBuilder()
            .add_species(glycerol_gas)
            .set_temperature(25+273.15, temperature_units="K")
            .set_pressure(1e5, pressure_units="Pa")
            .build()
        )

        # Glycerol particle distribution
        self.particle = (
            PresetParticleRadiusBuilder()
            .set_mode(np.array([100])*1e-9, "m")
            .set_geometric_standard_deviation(np.array([1.5]))
            .set_number_concentration(np.array([1e4])*1e-6, "1/m^3")
            .set_density(1.26e3, "kg/m^3")
            .build()
        )

        self.aerosol = Aerosol(
            atmosphere=self.atmosphere, particles=self.particle
        )

    def test_str(self):
        """Test the __str__ method of the Aerosol class."""
        result = str(self.aerosol)
        self.assertIn("Gas mixture", result)
        self.assertIn("Particle Representation", result)

    def test_iterate_gas(self):
        """Test the iterate_gas method of the Aerosol class."""
        gas_iterator = self.aerosol.iterate_gas()
        self.assertIsInstance(next(gas_iterator), GasSpecies)

    def test_iterate_particle(self):
        """Test the iterate_particle method of the Aerosol class."""
        particle_iterator = self.aerosol.iterate_particle()
        self.assertIsInstance(next(particle_iterator), ParticleRepresentation)

    def test_replace_atmosphere(self):
        """Test the replace_atmosphere method of the Aerosol class."""
        new_atmosphere = self.atmosphere
        self.aerosol.replace_atmosphere(new_atmosphere)
        self.assertEqual(self.aerosol.atmosphere, new_atmosphere)

    def test_add_particle(self):
        """Test the add_particle method of the Aerosol class."""
        new_particle = self.particle
        initial_count = len(self.aerosol.particles)
        self.aerosol.add_particle(new_particle)
        self.assertEqual(len(self.aerosol.particles), initial_count + 1)
