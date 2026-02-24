"""Unit tests for the ChargedCoagulationStrategy class.

This module contains tests for the ChargedCoagulationStrategy class, which
implements the charged particle coagulation strategy. The tests cover both
discrete and continuous_pdf distribution types.
"""

# pylint: disable=duplicate-code, too-many-instance-attributes

import unittest

import numpy as np
import pytest
from particula.dynamics.coagulation import charged_dimensional_kernel
from particula.dynamics.coagulation.charged_kernel_strategy import (
    HardSphereKernelStrategy,
)
from particula.dynamics.coagulation.coagulation_strategy.charged_coagulation_strategy import (  # noqa: E501
    ChargedCoagulationStrategy,
)
from particula.particles import (
    PresetParticleRadiusBuilder,
    PresetResolvedParticleMassBuilder,
    ResolvedParticleMassRepresentationBuilder,
)
from particula.particles.activity_strategies import ActivityIdealMass
from particula.particles.distribution_strategies import (
    ParticleResolvedSpeciatedMass,
)
from particula.particles.representation import ParticleRepresentation
from particula.particles.surface_strategies import SurfaceStrategyVolume

TEMPERATURE = 298.15
PRESSURE = 101325.0


def _build_particle_resolved_with_charges() -> ParticleRepresentation:
    radii = np.array(
        [50e-9, 55e-9, 5e-9, 6e-9],
        dtype=np.float64,
    )
    density = 2000.0
    mass = 4.0 / 3.0 * np.pi * radii**3 * density
    charges = np.array([-6.0, -6.0, 6.0, 6.0], dtype=np.float64)
    builder = ResolvedParticleMassRepresentationBuilder()
    builder.set_distribution_strategy(ParticleResolvedSpeciatedMass())
    builder.set_activity_strategy(ActivityIdealMass())
    builder.set_surface_strategy(SurfaceStrategyVolume())
    builder.set_mass(mass, "kg")
    builder.set_density(np.array([density], dtype=np.float64), "kg/m^3")
    builder.set_charge(charges)
    builder.set_volume(1e-15, "m^3")
    return builder.build()


def _build_two_particle_opposite_sign() -> ParticleRepresentation:
    density = np.array([2000.0, 2000.0], dtype=np.float64)
    radii = np.array([5e-9, 50e-9], dtype=np.float64)
    mass = 4.0 / 3.0 * np.pi * radii**3 * density
    builder = ResolvedParticleMassRepresentationBuilder()
    builder.set_distribution_strategy(ParticleResolvedSpeciatedMass())
    builder.set_activity_strategy(ActivityIdealMass())
    builder.set_surface_strategy(SurfaceStrategyVolume())
    builder.set_mass(mass, "kg")
    builder.set_density(density, "kg/m^3")
    builder.set_charge(np.array([6.0, -6.0], dtype=np.float64))
    builder.set_volume(1e-15, "m^3")
    return builder.build()


def _capture_collision_indices(
    particle: ParticleRepresentation,
    strategy: ChargedCoagulationStrategy,
    time_step: float,
    seed: int,
    volume: float | None,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Run a particle-resolved step while capturing collision indices."""
    collisions: dict[str, np.ndarray] = {}
    original_collide_pairs = particle.collide_pairs
    original_volume = particle.volume

    def _wrapped(indices: np.ndarray) -> None:
        collisions["indices"] = indices.copy()
        captured_charge = particle.get_charge()
        if captured_charge is not None:
            collisions["charges"] = captured_charge.copy()
        original_collide_pairs(indices)

    monkeypatch.setattr(particle, "collide_pairs", _wrapped)
    if volume is not None:
        particle.volume = volume
    strategy.random_generator = np.random.default_rng(seed=seed)
    strategy.step(
        particle=particle,
        temperature=TEMPERATURE,
        pressure=PRESSURE,
        time_step=time_step,
    )
    particle.volume = original_volume
    return (
        collisions.get("indices", np.empty((0, 2), dtype=np.int64)),
        collisions.get("charges"),
    )


class TestChargedCoagulationStrategy(unittest.TestCase):
    """Test suite for the ChargedCoagulationStrategy class."""

    def setUp(self):
        """Set up the test environment.

        Initializes a particle representation and creates instances of
        ChargedCoagulationStrategy for discrete, continuous_pdf, and
        particle_resolved distribution types.
        """
        # Setup a particle representation for testing
        self.particle = PresetParticleRadiusBuilder().build()
        self.temperature = TEMPERATURE
        self.pressure = PRESSURE

        # Create a kernel strategy instance
        self.kernel_strategy = HardSphereKernelStrategy()

        # Create strategies for all distribution types
        self.strategy_discrete = ChargedCoagulationStrategy(
            distribution_type="discrete",
            kernel_strategy=self.kernel_strategy,
        )
        self.strategy_continuous_pdf = ChargedCoagulationStrategy(
            distribution_type="continuous_pdf",
            kernel_strategy=self.kernel_strategy,
        )
        builder = PresetResolvedParticleMassBuilder()
        self.particle_resolved = builder.set_volume(1e-6, "m^3").build()
        self.strategy_particle_resolved = ChargedCoagulationStrategy(
            distribution_type="particle_resolved",
            kernel_strategy=self.kernel_strategy,
        )

    def test_kernel_discrete(self):
        """Test the kernel calculation for discrete distribution.

        Verifies that the kernel method returns an ndarray for the discrete
        distribution type.
        """
        # Test the kernel calculation for discrete distribution
        kernel = self.strategy_discrete.kernel(
            particle=self.particle,
            temperature=self.temperature,
            pressure=self.pressure,
        )
        self.assertIsInstance(kernel, np.ndarray)

    def test_step_discrete(self):
        """Test the step method for discrete distribution.

        Ensures that the step method updates the particle concentration for
        the discrete distribution type.
        """
        # Test the step method for discrete distribution
        initial_concentration = self.particle.get_concentration().copy()
        self.strategy_discrete.step(
            particle=self.particle,
            temperature=self.temperature,
            pressure=self.pressure,
            time_step=1.0,
        )
        updated_concentration = self.particle.get_concentration()
        self.assertFalse(
            np.array_equal(initial_concentration, updated_concentration)
        )

    def test_step_particle_resolved(self):
        """Test the kernel calculation for particle_resolved distribution."""
        # Test the kernel calculation for particle_resolved distribution
        old_concentration = self.particle_resolved.get_total_concentration()
        self.strategy_particle_resolved.step(
            particle=self.particle_resolved,
            temperature=self.temperature,
            pressure=self.pressure,
            time_step=1000,
        )
        new_concentration = self.particle_resolved.get_total_concentration()
        self.assertNotEqual(old_concentration, new_concentration)

    def test_kernel_continuous_pdf(self):
        """Test the kernel calculation for continuous_pdf distribution.

        Verifies that the kernel method returns an ndarray for the
        continuous_pdf distribution type.
        """
        # Test the kernel calculation for continuous_pdf distribution
        kernel = self.strategy_continuous_pdf.kernel(
            particle=self.particle,
            temperature=self.temperature,
            pressure=self.pressure,
        )
        self.assertIsInstance(kernel, np.ndarray)

    def test_step_continuous_pdf(self):
        """Test the step method for continuous_pdf distribution.

        Ensures that the step method updates the particle concentration for
        the continuous_pdf distribution type.
        """
        # Test the step method for continuous_pdf distribution
        initial_concentration = self.particle.get_concentration().copy()
        self.strategy_continuous_pdf.step(
            particle=self.particle,
            temperature=self.temperature,
            pressure=self.pressure,
            time_step=1.0,
        )
        updated_concentration = self.particle.get_concentration()
        self.assertFalse(
            np.array_equal(initial_concentration, updated_concentration)
        )


def test_direct_kernel_flag_default_false():
    """Default direct kernel flag should be disabled."""
    strategy = ChargedCoagulationStrategy(
        distribution_type="discrete",
        kernel_strategy=HardSphereKernelStrategy(),
    )
    assert strategy.use_direct_kernel is False


def test_direct_kernel_same_sign_no_spurious_mergers(monkeypatch):
    """Direct kernel should prevent same-sign coagulation artifacts."""
    particle = _build_particle_resolved_with_charges()
    charges = particle.get_charge()
    assert charges is not None

    def _selective_kernel(
        particle_radius: np.ndarray,
        particle_mass: np.ndarray,
        particle_charge: np.ndarray,
        temperature: float,
        pressure: float,
    ) -> np.ndarray:
        _ = particle_radius, particle_mass, temperature, pressure
        sign_product = np.sign(particle_charge[0]) * np.sign(particle_charge[1])
        # Zero for same-sign, large value for opposite-sign interactions
        kernel_value = 0.0 if sign_product >= 0 else 1e-6
        return np.array([[0.0, kernel_value], [kernel_value, 0.0]])

    monkeypatch.setattr(
        charged_dimensional_kernel,
        "get_hard_sphere_kernel_via_system_state",
        _selective_kernel,
    )

    strategy = ChargedCoagulationStrategy(
        distribution_type="particle_resolved",
        kernel_strategy=HardSphereKernelStrategy(),
        use_direct_kernel=True,
    )
    loss_gain_index, captured_charge = _capture_collision_indices(
        particle=particle,
        strategy=strategy,
        time_step=1.0,
        seed=10,
        volume=1e-12,
        monkeypatch=monkeypatch,
    )

    assert loss_gain_index.size > 0
    charge_snapshot = (
        captured_charge if captured_charge is not None else charges
    )
    pair_product = (
        charge_snapshot[loss_gain_index[:, 0]]
        * charge_snapshot[loss_gain_index[:, 1]]
    )
    assert np.all(pair_product <= 0)


def test_direct_kernel_opposite_sign_attracts(monkeypatch):
    """Direct kernel should still allow opposite-sign coagulation."""
    particle = _build_two_particle_opposite_sign()
    charges = particle.get_charge()
    assert charges is not None
    strategy = ChargedCoagulationStrategy(
        distribution_type="particle_resolved",
        kernel_strategy=HardSphereKernelStrategy(),
        use_direct_kernel=True,
    )
    loss_gain_index, captured_charge = _capture_collision_indices(
        particle=particle,
        strategy=strategy,
        time_step=10.0,
        seed=12,
        volume=None,
        monkeypatch=monkeypatch,
    )
    assert loss_gain_index.size > 0
    charge_snapshot = (
        captured_charge if captured_charge is not None else charges
    )
    small_index = loss_gain_index[:, 0]
    large_index = loss_gain_index[:, 1]
    opposite_sign = (
        (charge_snapshot[small_index] < 0) & (charge_snapshot[large_index] > 0)
    ) | (
        (charge_snapshot[small_index] > 0) & (charge_snapshot[large_index] < 0)
    )
    assert np.any(opposite_sign)


def test_direct_kernel_neutral_charge_defaults_to_zero(monkeypatch):
    """Direct kernel should handle particle-resolved cases without charge."""
    builder = PresetResolvedParticleMassBuilder()
    particle = builder.set_volume(1e-6, "m^3").build()
    monkeypatch.setattr(particle, "get_charge", lambda clone=False: None)

    from particula.dynamics.coagulation.coagulation_strategy import (
        coagulation_strategy_abc,
    )

    def _neutral_coulomb(
        particle_radius: np.ndarray,
        charge: np.ndarray | int | None = None,
        temperature: float = 298.15,
        ratio_lower_limit: float = -200,
    ) -> np.ndarray:
        _ = charge, temperature, ratio_lower_limit
        count = len(particle_radius)
        return np.zeros((count, count), dtype=np.float64)

    monkeypatch.setattr(
        coagulation_strategy_abc.particles,
        "get_coulomb_enhancement_ratio",
        _neutral_coulomb,
    )

    strategy = ChargedCoagulationStrategy(
        distribution_type="particle_resolved",
        kernel_strategy=HardSphereKernelStrategy(),
        use_direct_kernel=True,
    )
    strategy.random_generator = np.random.default_rng(seed=21)
    strategy.step(
        particle=particle,
        temperature=TEMPERATURE,
        pressure=PRESSURE,
        time_step=1.0,
    )

    assert particle.get_charge() is None
