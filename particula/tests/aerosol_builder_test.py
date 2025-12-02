"""Unit tests verifying that `AerosolBuilder._validate_species_length`
correctly checks consistency between gas-phase and particle-phase
species counts for different scenarios.
"""

# pylint: disable=W0212, R0801

import copy
import unittest

import numpy as np

from particula.aerosol import Aerosol
from particula.aerosol_builder import AerosolBuilder
from particula.gas import (
    AtmosphereBuilder,
    GasSpeciesBuilder,
    VaporPressureFactory,
)
from particula.particles import (
    PresetResolvedParticleMassBuilder,
)


def _build_atmosphere(n_species: int):
    """Return an Atmosphere containing *n_species* identical
    partitioning vapours for use in the validation tests.
    """
    molar_mass = 92.09382e-3  # kg/mol (glycerol)
    params = {
        "latent_heat": 71.5e3 * molar_mass,
        "latent_heat_units": "J/mol",
        "temperature_initial": 125.5 + 273.15,
        "temperature_initial_units": "K",
        "pressure_initial": 1e5,
        "pressure_initial_units": "Pa",
    }
    vp_strategy = VaporPressureFactory().get_strategy(
        "clausius_clapeyron", params
    )
    sat_conc = vp_strategy.saturation_concentration(molar_mass, 298.15)
    gas = (
        GasSpeciesBuilder()
        .set_molar_mass(molar_mass, "kg/mol")
        .set_vapor_pressure_strategy(vp_strategy)
        .set_concentration(0.5 * sat_conc, "kg/m^3")
        .set_name("Glycerol")
        .set_partitioning(True)
        .build()
    )
    atm_builder = (
        AtmosphereBuilder()
        .set_temperature(25 + 273.15, temperature_units="K")
        .set_pressure(1e5, pressure_units="Pa")
    )
    for _ in range(n_species):
        atm_builder = atm_builder.set_more_partitioning_species(
            copy.deepcopy(gas)
        )
    return atm_builder.build()


def _build_particle():
    """Return a minimal single-species particle representation
    built with `PresetResolvedParticleMassBuilder`.
    """
    return PresetResolvedParticleMassBuilder().build()


def _speciated_clone(particle, n_species: int):
    """Deep-copy *particle* and tile its distribution to mimic a
    `SpeciatedMassMovingBin` object with *n_species* particle-phase
    species.
    """
    particle = copy.deepcopy(particle)

    distribution_org = particle.get_distribution(clone=True)
    # repeat the distribution *n_species* times
    # We tile the original distribution to simulate multiple species because
    # directly instantiating a distribution for each species is challenging.
    # This tiling ensures that the format conforms with SpeciatedMassMovingBin
    # and ParticleResolvedSpeciatedMass, especially in how get_species_mass
    # extracts species counts from the structured (tiled) distribution.
    distribution = np.tile(distribution_org, (n_species, 1)).transpose()
    particle.distribution = distribution
    return particle


class TestAerosolBuilderValidation(unittest.TestCase):
    """Test suite for `AerosolBuilder` species-count validation."""

    def setUp(self):
        """Create a reusable single-species particle for the tests."""
        # one reusable base particle
        self.base_particle = _build_particle()

    # ---------- NO GAS SPECIES CASE ----------
    def test_no_species_passes(self):
        """Validation passes when no gas species are present."""
        atm = _build_atmosphere(0)
        pr = _speciated_clone(self.base_particle, 1)
        AerosolBuilder().set_atmosphere(atm).set_particles(
            pr
        )._validate_species_length()

    # ---------- SINGLE-SPECIES CASE ----------
    def test_single_species_match_passes(self):
        """Validation passes when one gas species matches one particle
        species.
        """
        atm = _build_atmosphere(1)
        pr = _speciated_clone(self.base_particle, 1)
        AerosolBuilder().set_atmosphere(atm).set_particles(
            pr
        )._validate_species_length()

    def test_single_species_mismatch_raises(self):
        """Validation fails when one gas species does NOT match two particle
        species.
        """
        atm = _build_atmosphere(1)
        pr = _speciated_clone(self.base_particle, 2)
        with self.assertRaises(ValueError):
            AerosolBuilder().set_atmosphere(atm).set_particles(
                pr
            )._validate_species_length()

    # ---------- THREE-SPECIES CASE ----------
    def test_three_species_match_passes(self):
        """Validation passes when three gas species match three particle
        species.
        """
        atm = _build_atmosphere(3)
        pr = _speciated_clone(self.base_particle, 3)
        AerosolBuilder().set_atmosphere(atm).set_particles(
            pr
        )._validate_species_length()

    def test_three_species_mismatch_raises(self):
        """Validation fails when three gas species do NOT match two particle
        species.
        """
        atm = _build_atmosphere(3)
        pr = _speciated_clone(self.base_particle, 2)
        with self.assertRaises(ValueError):
            AerosolBuilder().set_atmosphere(atm).set_particles(
                pr
            )._validate_species_length()

    def test_full_build_returns_aerosol(self):
        """Ensure a fully-configured builder returns an Aerosol instance."""
        atm = _build_atmosphere(2)  # 2 partitioning species
        pr = _speciated_clone(self.base_particle, 2)  # 2 particle species
        aerosol = AerosolBuilder().set_atmosphere(atm).set_particles(pr).build()
        self.assertIsInstance(aerosol, Aerosol)
