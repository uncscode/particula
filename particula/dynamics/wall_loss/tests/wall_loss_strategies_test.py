"""Tests for wall loss strategy classes.

Covers the abstract :class:`WallLossStrategy` interface and the
concrete :class:`SphericalWallLossStrategy` implementation, as well as
the :func:`get_particle_resolved_wall_loss_step` helper function.
"""

import unittest

import numpy as np

from particula.dynamics.wall_loss.wall_loss_strategies import (
    SphericalWallLossStrategy,
    WallLossStrategy,
    get_particle_resolved_wall_loss_step,
)
from particula.particles import (
    PresetParticleRadiusBuilder,
    PresetResolvedParticleMassBuilder,
)


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


class TestGetParticleResolvedWallLossStep(unittest.TestCase):
    """Test suite for get_particle_resolved_wall_loss_step function."""

    def setUp(self):
        """Set up test fixtures."""
        self.rng = np.random.default_rng(42)

    def _constant_coeff_func(self, coefficient: float):
        """Return a coefficient function that returns a constant value."""

        def coeff_func(radius, density):
            return coefficient * np.ones_like(radius)

        return coeff_func

    def test_returns_array_of_correct_shape(self):
        """Returned survival array matches input concentration shape."""
        radius = np.array([1e-7, 2e-7, 3e-7])
        density = np.array([1000.0, 1000.0, 1000.0])
        concentration = np.array([1e6, 1e6, 1e6])

        survived = get_particle_resolved_wall_loss_step(
            particle_radius=radius,
            particle_density=density,
            concentration=concentration,
            loss_coefficient_func=self._constant_coeff_func(1e-4),
            time_step=1.0,
            random_generator=self.rng,
        )

        self.assertEqual(survived.shape, concentration.shape)

    def test_survival_values_are_binary(self):
        """Survival values should be either 0 or 1."""
        radius = np.array([1e-7, 2e-7, 3e-7, 4e-7, 5e-7])
        density = np.array([1000.0] * 5)
        concentration = np.array([1e6] * 5)

        survived = get_particle_resolved_wall_loss_step(
            particle_radius=radius,
            particle_density=density,
            concentration=concentration,
            loss_coefficient_func=self._constant_coeff_func(0.5),
            time_step=1.0,
            random_generator=self.rng,
        )

        self.assertTrue(np.all((survived == 0.0) | (survived == 1.0)))

    def test_zero_coefficient_all_survive(self):
        """With zero loss coefficient, all particles should survive."""
        radius = np.array([1e-7, 2e-7, 3e-7])
        density = np.array([1000.0, 1000.0, 1000.0])
        concentration = np.array([1e6, 1e6, 1e6])

        survived = get_particle_resolved_wall_loss_step(
            particle_radius=radius,
            particle_density=density,
            concentration=concentration,
            loss_coefficient_func=self._constant_coeff_func(0.0),
            time_step=1.0,
            random_generator=self.rng,
        )

        self.assertTrue(np.all(survived == 1.0))

    def test_very_high_coefficient_most_particles_lost(self):
        """With very high loss coefficient, most particles should be lost."""
        n_particles = 1000
        radius = np.full(n_particles, 1e-7)
        density = np.full(n_particles, 1000.0)
        concentration = np.full(n_particles, 1e6)

        survived = get_particle_resolved_wall_loss_step(
            particle_radius=radius,
            particle_density=density,
            concentration=concentration,
            loss_coefficient_func=self._constant_coeff_func(100.0),
            time_step=1.0,
            random_generator=self.rng,
        )

        # With coefficient=100 and dt=1, survival prob = exp(-100) ~ 0
        survival_fraction = np.sum(survived) / n_particles
        self.assertLess(survival_fraction, 0.01)

    def test_zero_radius_particles_return_zero_survival(self):
        """Particles with zero radius should have zero survival."""
        radius = np.array([0.0, 1e-7, 0.0, 2e-7])
        density = np.array([1000.0, 1000.0, 1000.0, 1000.0])
        concentration = np.array([1e6, 1e6, 1e6, 1e6])

        survived = get_particle_resolved_wall_loss_step(
            particle_radius=radius,
            particle_density=density,
            concentration=concentration,
            loss_coefficient_func=self._constant_coeff_func(0.0),
            time_step=1.0,
            random_generator=self.rng,
        )

        # Zero radius particles should not survive
        self.assertEqual(survived[0], 0.0)
        self.assertEqual(survived[2], 0.0)
        # Non-zero radius with zero coeff should survive
        self.assertEqual(survived[1], 1.0)
        self.assertEqual(survived[3], 1.0)

    def test_all_zero_radius_returns_all_zeros(self):
        """If all particles have zero radius, return all zeros."""
        radius = np.array([0.0, 0.0, 0.0])
        density = np.array([1000.0, 1000.0, 1000.0])
        concentration = np.array([1e6, 1e6, 1e6])

        survived = get_particle_resolved_wall_loss_step(
            particle_radius=radius,
            particle_density=density,
            concentration=concentration,
            loss_coefficient_func=self._constant_coeff_func(0.0),
            time_step=1.0,
            random_generator=self.rng,
        )

        self.assertTrue(np.all(survived == 0.0))

    def test_stochastic_behavior_different_seeds(self):
        """Different random seeds should produce different results."""
        n_particles = 100
        radius = np.full(n_particles, 1e-7)
        density = np.full(n_particles, 1000.0)
        concentration = np.full(n_particles, 1e6)

        # Use moderate coefficient for ~50% survival probability
        coeff_func = self._constant_coeff_func(0.7)

        rng1 = np.random.default_rng(123)
        survived1 = get_particle_resolved_wall_loss_step(
            particle_radius=radius,
            particle_density=density,
            concentration=concentration,
            loss_coefficient_func=coeff_func,
            time_step=1.0,
            random_generator=rng1,
        )

        rng2 = np.random.default_rng(456)
        survived2 = get_particle_resolved_wall_loss_step(
            particle_radius=radius,
            particle_density=density,
            concentration=concentration,
            loss_coefficient_func=coeff_func,
            time_step=1.0,
            random_generator=rng2,
        )

        # Results should differ due to different seeds
        self.assertFalse(np.array_equal(survived1, survived2))

    def test_reproducible_with_same_seed(self):
        """Same random seed should produce identical results."""
        radius = np.array([1e-7, 2e-7, 3e-7, 4e-7, 5e-7])
        density = np.array([1000.0] * 5)
        concentration = np.array([1e6] * 5)
        coeff_func = self._constant_coeff_func(0.5)

        rng1 = np.random.default_rng(42)
        survived1 = get_particle_resolved_wall_loss_step(
            particle_radius=radius,
            particle_density=density,
            concentration=concentration,
            loss_coefficient_func=coeff_func,
            time_step=1.0,
            random_generator=rng1,
        )

        rng2 = np.random.default_rng(42)
        survived2 = get_particle_resolved_wall_loss_step(
            particle_radius=radius,
            particle_density=density,
            concentration=concentration,
            loss_coefficient_func=coeff_func,
            time_step=1.0,
            random_generator=rng2,
        )

        np.testing.assert_array_equal(survived1, survived2)

    def test_survival_probability_statistical(self):
        """Survival fraction should match expected probability statistically."""
        n_particles = 10000
        radius = np.full(n_particles, 1e-7)
        density = np.full(n_particles, 1000.0)
        concentration = np.full(n_particles, 1e6)

        # coefficient=1.0, dt=1.0 -> survival_prob = exp(-1) ~ 0.368
        expected_survival = np.exp(-1.0)
        coeff_func = self._constant_coeff_func(1.0)

        survived = get_particle_resolved_wall_loss_step(
            particle_radius=radius,
            particle_density=density,
            concentration=concentration,
            loss_coefficient_func=coeff_func,
            time_step=1.0,
            random_generator=self.rng,
        )

        actual_survival = np.sum(survived) / n_particles
        # Allow 5% tolerance for statistical variation
        self.assertAlmostEqual(actual_survival, expected_survival, delta=0.05)

    def test_time_step_affects_survival(self):
        """Longer time step should result in fewer survivors."""
        n_particles = 1000
        radius = np.full(n_particles, 1e-7)
        density = np.full(n_particles, 1000.0)
        concentration = np.full(n_particles, 1e6)
        coeff_func = self._constant_coeff_func(1.0)

        rng1 = np.random.default_rng(42)
        survived_short = get_particle_resolved_wall_loss_step(
            particle_radius=radius,
            particle_density=density,
            concentration=concentration,
            loss_coefficient_func=coeff_func,
            time_step=0.1,
            random_generator=rng1,
        )

        rng2 = np.random.default_rng(42)
        survived_long = get_particle_resolved_wall_loss_step(
            particle_radius=radius,
            particle_density=density,
            concentration=concentration,
            loss_coefficient_func=coeff_func,
            time_step=10.0,
            random_generator=rng2,
        )

        # Longer time step should have fewer survivors
        self.assertGreater(np.sum(survived_short), np.sum(survived_long))
