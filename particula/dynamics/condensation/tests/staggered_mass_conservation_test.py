"""Mass conservation tests for CondensationIsothermalStaggered.

This suite validates that particle + gas mass remains conserved for the
staggered condensation strategy across theta modes, particle counts, step
counts, and large time steps. Tolerances follow issue requirements:
- Single step: 1e-12 relative (machine precision)
- Multi-step (10, 100, 1000 steps): 1e-10 relative (accumulated error
  allowance)
- Large time step: 1e-3 relative (relaxed due to numerical approximation
  limits at large dt)
Heavy cases (n_particles>=1000, steps>=100, or their combinations) are marked
with @pytest.mark.performance so the default suite stays fast.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from particula.dynamics.condensation import CondensationIsothermalStaggered
from particula.gas import GasSpeciesBuilder, VaporPressureFactory
from particula.particles import (
    ActivityIdealMass,
    ParticleResolvedSpeciatedMassBuilder,
    ResolvedParticleMassRepresentationBuilder,
    SurfaceStrategyVolume,
)


def calculate_total_mass(
    particle,
    gas_species,
) -> float:
    """Return total system mass (particle + gas) in kilograms.

    Args:
        particle: ParticleRepresentation with resolved mass distribution.
        gas_species: GasSpecies carrying condensing species concentration.

    Returns:
        Total mass in kilograms for conservation checks.
    """
    particle_mass = float(np.sum(particle.get_mass()))
    gas_mass = float(np.sum(gas_species.get_concentration()))
    return particle_mass + gas_mass


@pytest.fixture()
def create_test_system():
    """Factory fixture that builds deterministic particle/gas systems.

    Returns:
        Callable that produces (particle, gas_species) pairs for a given
        particle count and RNG seed. Mass arrays are shaped (n, 1) to match
        a single condensing species.
    """
    vp_strategy = VaporPressureFactory().get_strategy(
        "constant",
        {
            "vapor_pressure": 101325.0,
            "vapor_pressure_units": "Pa",
        },
    )

    def _create(n_particles: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        masses: NDArray[np.float64] = rng.lognormal(
            mean=-20.0, sigma=0.5, size=(n_particles, 1)
        )

        distribution_strategy = ParticleResolvedSpeciatedMassBuilder().build()
        particle = (
            ResolvedParticleMassRepresentationBuilder()
            .set_distribution_strategy(distribution_strategy)
            .set_activity_strategy(ActivityIdealMass())
            .set_surface_strategy(
                SurfaceStrategyVolume(surface_tension=0.072, density=1000.0)
            )
            .set_mass(masses, "kg")
            .set_density(np.full_like(masses, 1000.0), "kg/m^3")
            .set_charge(np.zeros_like(masses))
            .set_volume(1.0, "m^3")
            .build()
        )

        gas_species = (
            GasSpeciesBuilder()
            .set_name("water")
            .set_molar_mass(0.018, "kg/mol")
            .set_vapor_pressure_strategy(vp_strategy)
            .set_partitioning(True)
            .set_concentration(0.01, "kg/m^3")
            .build()
        )

        return particle, gas_species

    return _create


@pytest.fixture()
def small_particles():
    """Create sub-10 nm particles to stress Kelvin curvature effects.

    Note: Tests mutate the returned gas_species via set_concentration().
    The fixture uses default function scope, so each test receives a fresh
    instance to ensure isolation.
    """
    diameters = np.array([5e-9, 7e-9, 9e-9])
    density = 1000.0
    volumes = (4.0 / 3.0) * np.pi * (diameters / 2.0) ** 3
    masses = (volumes * density).reshape(-1, 1)

    distribution_strategy = ParticleResolvedSpeciatedMassBuilder().build()
    particle = (
        ResolvedParticleMassRepresentationBuilder()
        .set_distribution_strategy(distribution_strategy)
        .set_activity_strategy(ActivityIdealMass())
        .set_surface_strategy(
            SurfaceStrategyVolume(surface_tension=0.072, density=density)
        )
        .set_mass(masses, "kg")
        .set_density(np.full_like(masses, density), "kg/m^3")
        .set_charge(np.zeros_like(masses))
        .set_volume(1.0, "m^3")
        .build()
    )

    gas_species = (
        GasSpeciesBuilder()
        .set_name("water")
        .set_molar_mass(0.018, "kg/mol")
        .set_vapor_pressure_strategy(
            VaporPressureFactory().get_strategy(
                "constant",
                {
                    "vapor_pressure": 101325.0,
                    "vapor_pressure_units": "Pa",
                },
            )
        )
        .set_partitioning(True)
        .set_concentration(0.01, "kg/m^3")
        .build()
    )

    return particle, gas_species


class TestMassConservation:
    """Mass conservation checks across theta modes and step regimes."""

    # Single-step tolerance ensures machine precision.
    RELATIVE_TOLERANCE = 1e-12
    # Multi-step tolerance allows slight accumulation over up to 1000 steps.
    MULTI_STEP_TOLERANCE = 1e-10
    # Large time steps allow a relaxed tolerance for non-accumulating checks.
    LARGE_TIME_STEP_TOLERANCE = 1e-3

    @pytest.mark.parametrize(
        "n_particles",
        [
            1,
            100,
            pytest.param(1000, marks=pytest.mark.performance),
            pytest.param(10000, marks=pytest.mark.performance),
        ],
    )
    @pytest.mark.parametrize(
        "theta_mode",
        ["half", "random", "batch"],
    )
    def test_mass_conservation_single_step(
        self,
        theta_mode: str,
        n_particles: int,
        create_test_system,
    ) -> None:
        """Single step should conserve mass to 1e-12 relative tolerance."""
        particle, gas_species = create_test_system(n_particles)
        initial_mass = calculate_total_mass(particle, gas_species)

        strategy = CondensationIsothermalStaggered(
            molar_mass=0.018,
            theta_mode=theta_mode,
            num_batches=10,
            random_state=42,
            shuffle_each_step=False,
        )

        particle_new, gas_new = strategy.step(
            particle,
            gas_species,
            298.0,
            101325.0,
            time_step=0.0002,
        )

        final_mass = calculate_total_mass(particle_new, gas_new)
        relative_error = abs(final_mass - initial_mass) / initial_mass
        assert relative_error < self.RELATIVE_TOLERANCE

    @pytest.mark.parametrize(
        ("steps", "n_particles"),
        [
            (10, 100),
            pytest.param(10, 1000, marks=pytest.mark.performance),
            pytest.param(100, 100, marks=pytest.mark.performance),
            pytest.param(100, 1000, marks=pytest.mark.performance),
        ],
    )
    # Batch mode excluded: deterministic batching with fixed RNG already
    # exercised in single-step and large time step tests; random mode provides
    # better coverage of stochastic accumulation in multi-step scenarios.
    @pytest.mark.parametrize(
        "theta_mode",
        ["half", "random"],
    )
    def test_mass_conservation_multi_step(
        self,
        theta_mode: str,
        steps: int,
        n_particles: int,
        create_test_system,
    ) -> None:
        """Multi-step paths allow 1e-10 tolerance with bounded accumulation."""
        particle, gas_species = create_test_system(n_particles)
        initial_mass = calculate_total_mass(particle, gas_species)

        strategy = CondensationIsothermalStaggered(
            molar_mass=0.018,
            theta_mode=theta_mode,
            num_batches=10,
            random_state=42,
            shuffle_each_step=False,
        )

        # Use shorter time steps to keep accumulation within tolerance.
        # Smaller dt for longer runs reduces per-step numerical error,
        # maintaining cumulative error below MULTI_STEP_TOLERANCE.
        time_step_short = 0.0002  # For 10-100 step runs
        time_step_long = 0.00005  # For 100+ step runs
        time_step = time_step_short if steps < 100 else time_step_long
        for _ in range(steps):
            particle, gas_species = strategy.step(
                particle,
                gas_species,
                298.0,
                101325.0,
                time_step=time_step,
            )

        final_mass = calculate_total_mass(particle, gas_species)
        relative_error = abs(final_mass - initial_mass) / initial_mass
        assert relative_error < self.MULTI_STEP_TOLERANCE

    @pytest.mark.performance
    def test_mass_conservation_multi_step_slow(
        self,
        create_test_system,
    ) -> None:
        """Long-horizon multi-step run stays within the standard tolerance."""
        n_particles = 100
        steps = 1000
        particle, gas_species = create_test_system(n_particles)
        initial_mass = calculate_total_mass(particle, gas_species)

        strategy = CondensationIsothermalStaggered(
            molar_mass=0.018,
            theta_mode="half",
            num_batches=10,
            random_state=42,
            shuffle_each_step=False,
        )

        # Use an even smaller time step for the long-horizon slow path.
        time_step = 0.00005
        for _ in range(steps):
            particle, gas_species = strategy.step(
                particle,
                gas_species,
                298.0,
                101325.0,
                time_step=time_step,
            )

        final_mass = calculate_total_mass(particle, gas_species)
        relative_error = abs(final_mass - initial_mass) / initial_mass
        assert relative_error < self.MULTI_STEP_TOLERANCE

    @pytest.mark.parametrize("theta_mode", ["batch", "half"])
    @pytest.mark.parametrize("time_step", [1.0, 10.0, 100.0])
    def test_mass_conservation_large_timestep(
        self,
        theta_mode: str,
        time_step: float,
        create_test_system,
    ) -> None:
        """Large time steps still conserve mass to ~1e-3 relative for
        deterministic theta modes.
        """
        particle, gas_species = create_test_system(500)
        initial_mass = calculate_total_mass(particle, gas_species)

        strategy = CondensationIsothermalStaggered(
            molar_mass=0.018,
            theta_mode=theta_mode,
            num_batches=10,
            random_state=42,
            shuffle_each_step=False,
        )

        particle_new, gas_new = strategy.step(
            particle,
            gas_species,
            298.0,
            101325.0,
            time_step=time_step,
        )

        final_mass = calculate_total_mass(particle_new, gas_new)
        relative_error = abs(final_mass - initial_mass) / initial_mass
        assert relative_error < self.LARGE_TIME_STEP_TOLERANCE


class TestKelvinEffectConservation:
    """Stress Kelvin curvature cases while conserving total mass.

    These scenarios use sub-10 nm particles where curvature raises equilibrium
    vapor pressure (Kelvin effect). Each test configures explicit super- or
    subsaturation to drive condensation or evaporation and asserts total
    particle + gas mass is conserved to 1e-12 relative tolerance.
    """

    RELATIVE_TOLERANCE = 1e-12
    DEFAULT_TIME_STEP = 0.001
    EQUILIBRIUM_MASS_DRIFT_THRESHOLD = 1e-15

    @staticmethod
    def _strategy(theta_mode: str = "half", num_batches: int = 3):
        return CondensationIsothermalStaggered(
            molar_mass=0.018,
            theta_mode=theta_mode,
            num_batches=num_batches,
            random_state=42,
            shuffle_each_step=False,
        )

    def test_kelvin_small_particles_evaporation(self, small_particles) -> None:
        """Strong subsaturation drives Kelvin-enhanced evaporation."""
        particle, gas_species = small_particles
        # Push the vapor well below equilibrium to force evaporation.
        gas_species.set_concentration(1e-4)

        initial_mass = calculate_total_mass(particle, gas_species)
        strategy = self._strategy(theta_mode="half")
        particle_new, gas_new = strategy.step(
            particle,
            gas_species,
            298.0,
            101325.0,
            time_step=self.DEFAULT_TIME_STEP,
        )

        final_mass = calculate_total_mass(particle_new, gas_new)
        relative_error = abs(final_mass - initial_mass) / initial_mass
        assert relative_error < self.RELATIVE_TOLERANCE, (
            "Kelvin evaporation violated mass conservation; "
            f"relative error={relative_error:.2e}"
        )

    def test_kelvin_supersaturation_condensation(self, small_particles) -> None:
        """Supersaturation forces condensation while conserving mass."""
        particle, gas_species = small_particles
        # Five times the baseline concentration yields clear supersaturation.
        gas_species.set_concentration(5e-2)

        initial_mass = calculate_total_mass(particle, gas_species)
        strategy = self._strategy(theta_mode="half")
        particle_new, gas_new = strategy.step(
            particle,
            gas_species,
            298.0,
            101325.0,
            time_step=self.DEFAULT_TIME_STEP,
        )

        final_mass = calculate_total_mass(particle_new, gas_new)
        relative_error = abs(final_mass - initial_mass) / initial_mass
        assert relative_error < self.RELATIVE_TOLERANCE, (
            "Supersaturation condensation violated mass conservation; "
            f"relative error={relative_error:.2e}"
        )

    def test_kelvin_subsaturation_evaporation(self, small_particles) -> None:
        """Mild subsaturation evaporates mass without loss of total mass."""
        particle, gas_species = small_particles
        # Lower concentration (20x below baseline) for moderate evaporation.
        gas_species.set_concentration(5e-4)

        initial_mass = calculate_total_mass(particle, gas_species)
        strategy = self._strategy(theta_mode="half")
        particle_new, gas_new = strategy.step(
            particle,
            gas_species,
            298.0,
            101325.0,
            time_step=self.DEFAULT_TIME_STEP,
        )

        final_mass = calculate_total_mass(particle_new, gas_new)
        relative_error = abs(final_mass - initial_mass) / initial_mass
        assert relative_error < self.RELATIVE_TOLERANCE, (
            "Subsaturation evaporation violated mass conservation; "
            f"relative error={relative_error:.2e}"
        )

    def test_kelvin_mixed_supersaturation_subsaturation(
        self, small_particles
    ) -> None:
        """Mixed conditions shrink smallest particles and grow largest."""
        particle, gas_species = small_particles
        # Mid-range vapor lets Kelvin curvature split growth vs evaporation.
        gas_species.set_concentration(1.0)
        initial_mass = calculate_total_mass(particle, gas_species)
        initial_particle_mass = np.array(particle.get_mass(), copy=True)

        strategy = self._strategy(theta_mode="batch", num_batches=3)
        particle_new, gas_new = strategy.step(
            particle,
            gas_species,
            298.0,
            101325.0,
            time_step=0.0005,
        )

        final_mass = calculate_total_mass(particle_new, gas_new)
        relative_error = abs(final_mass - initial_mass) / initial_mass
        masses_new = np.array(particle_new.get_mass())
        mass_change = masses_new - initial_particle_mass
        assert np.any(mass_change < 0.0)
        assert np.any(mass_change > 0.0)
        assert relative_error < self.RELATIVE_TOLERANCE, (
            "Mixed saturation case lost mass; "
            f"relative error={relative_error:.2e}"
        )

    def test_kelvin_critical_diameter_behavior(self, small_particles) -> None:
        """Near-equilibrium vapor keeps mass conserved with minimal drift."""
        particle, gas_species = small_particles
        gas_species.set_concentration(0.01)

        initial_mass = calculate_total_mass(particle, gas_species)
        strategy = self._strategy(theta_mode="half")
        particle_new, gas_new = strategy.step(
            particle,
            gas_species,
            298.0,
            101325.0,
            time_step=0.0005,
        )

        final_mass = calculate_total_mass(particle_new, gas_new)
        relative_error = abs(final_mass - initial_mass) / initial_mass
        # Stay close to equilibrium: per-particle mass drift is tiny.
        mass_delta = np.linalg.norm(
            particle_new.get_mass() - particle.get_mass()
        )
        assert mass_delta < self.EQUILIBRIUM_MASS_DRIFT_THRESHOLD
        assert relative_error < self.RELATIVE_TOLERANCE, (
            "Critical diameter case drifted mass; "
            f"relative error={relative_error:.2e}"
        )
