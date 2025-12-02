"""Test the change of particle representation."""

import unittest

import numpy as np

from particula.particles import PresetResolvedParticleMassBuilder
from particula.particles.change_particle_representation import (
    get_particle_resolved_binned_radius,
    get_speciated_mass_representation_from_particle_resolved,
)
from particula.particles.representation import ParticleRepresentation


class TestChangeParticleRepresentation(unittest.TestCase):
    """Test suite for the change_particle_representation module."""

    def setUp(self):
        """Set up the test environment.

        Initializes a particle representation using the
        PresetResolvedParticleMassBuilder.
        """
        self.particle = PresetResolvedParticleMassBuilder().build()

    def test_get_particle_resolved_binned_radius(self):
        """Test the get_particle_resolved_binned_radius function.

        Verifies that the function returns the correct binned radii.
        """
        bin_radius = get_particle_resolved_binned_radius(
            self.particle, total_bins=10
        )
        self.assertEqual(len(bin_radius), 10)
        self.assertTrue(np.all(bin_radius > 0))

    def test_get_speciated_mass_representation_single(self):
        """Test the get_speciated_mass_representation_from_particle_resolved
        function.

        Verifies that the function returns a new ParticleRepresentation with
        binned mass.
        """
        bin_radius = get_particle_resolved_binned_radius(
            self.particle, total_bins=10
        )
        new_particle = get_speciated_mass_representation_from_particle_resolved(
            self.particle, bin_radius
        )
        self.assertIsInstance(new_particle, ParticleRepresentation)
        self.assertTrue(np.all(new_particle.get_distribution() >= 0))

    def test_get_speciated_mass_representation_double(self):
        """Test the get_speciated_mass_representation_from_particle_resolved
        function.

        Verifies that the function returns a new ParticleRepresentation with
        binned mass.
        """
        bin_radius = get_particle_resolved_binned_radius(
            self.particle, total_bins=10
        )
        self.particle.distribution = np.column_stack(
            (
                self.particle.distribution,
                self.particle.distribution,
            )
        )
        new_particle = get_speciated_mass_representation_from_particle_resolved(
            self.particle, bin_radius
        )
        self.assertIsInstance(new_particle, ParticleRepresentation)
        self.assertTrue(np.all(new_particle.get_distribution() >= 0))
