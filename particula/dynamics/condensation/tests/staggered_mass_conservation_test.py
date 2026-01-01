"""Mass conservation tests for CondensationIsothermalStaggered.

This suite validates that particle + gas mass remains conserved for the
staggered condensation strategy across theta modes, particle counts, step
counts, and large time steps. Tolerances follow issue requirements:
- Single step: 1e-12 relative (machine precision)
- Multi-step (10, 100, 1000 steps): 1e-10 relative (accumulated error allowance)
- Large time step: 1e-3 relative (non-accumulating regimes)
Heavy cases (n_particles=10000 single step and the 1000-step multi-step
scenario) are marked slow so the default suite stays fast.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from _pytest.warning_types import PytestUnknownMarkWarning
from numpy.typing import NDArray

from particula.dynamics.condensation import CondensationIsothermalStaggered
from particula.gas import GasSpeciesBuilder, VaporPressureFactory
from particula.particles import (
    ActivityIdealMass,
    ParticleResolvedSpeciatedMassBuilder,
    ResolvedParticleMassRepresentationBuilder,
    SurfaceStrategyVolume,
)

warnings.filterwarnings("ignore", category=PytestUnknownMarkWarning)


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


class TestMassConservation:
    """Mass conservation checks across theta modes and step regimes."""

    # Single-step tolerance ensures machine precision.
    RELATIVE_TOLERANCE = 1e-12
    # Multi-step tolerance allows slight accumulation over O(1000) steps.
    MULTI_STEP_TOLERANCE = 1e-10
    # Large time steps allow a relaxed tolerance for non-accumulating checks.
    LARGE_TIME_STEP_TOLERANCE = 1e-3

    @pytest.mark.parametrize(
        "n_particles",
        [
            1,
            100,
            1000,
            pytest.param(10000, marks=pytest.mark.slow),
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
            time_step=0.001,
        )

        final_mass = calculate_total_mass(particle_new, gas_new)
        relative_error = abs(final_mass - initial_mass) / initial_mass
        assert relative_error < self.RELATIVE_TOLERANCE

    @pytest.mark.parametrize(
        ("steps", "n_particles"),
        [
            (10, 100),
            (10, 1000),
            pytest.param(100, 100, marks=pytest.mark.slow),
            pytest.param(100, 1000, marks=pytest.mark.slow),
        ],
    )
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
        time_step = 0.0002 if steps < 100 else 0.00005
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

    @pytest.mark.slow
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

    def test_mass_conservation_zero_time_step(
        self,
        create_test_system,
    ) -> None:
        """time_step=0 should leave masses unchanged (no-op path)."""
        particle, gas_species = create_test_system(5)
        initial_mass = calculate_total_mass(particle, gas_species)

        strategy = CondensationIsothermalStaggered(
            molar_mass=0.018,
            theta_mode="half",
            num_batches=10,
            random_state=42,
            shuffle_each_step=False,
        )

        particle_new, gas_new = strategy.step(
            particle,
            gas_species,
            298.0,
            101325.0,
            time_step=0.0,
        )

        final_mass = calculate_total_mass(particle_new, gas_new)
        np.testing.assert_allclose(initial_mass, final_mass, rtol=0.0, atol=0.0)
