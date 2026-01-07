# Feature E1-F4: Mass Conservation Validation

**Status:** Not Started
**Priority:** P2
**Assignees:** TBD
**Labels:** feature, dynamics, condensation, testing, validation
**Milestone:** v0.3.x
**Size:** S (~100 LOC tests)

**Start Date:** TBD
**Target Date:** TBD
**Created:** 2025-12-23
**Updated:** 2025-12-23

**Parent Epic:** [E1: Staggered ODE Stepping][epic-e1]
**Related Issues:** TBD
**Related PRs:** TBD

---

## Overview

Create comprehensive test suites that validate mass conservation properties of
the staggered condensation algorithm. These tests verify that total mass (gas +
particle) is preserved to within numerical tolerance across all stepping modes
and edge cases.

### Problem Statement

Mass conservation is the primary success criterion for the staggered stepping
algorithm. Without rigorous validation:

- Subtle numerical bugs could cause mass drift over long simulations
- Edge cases (small particles, supersaturation) might violate conservation
- Different theta modes might have different conservation properties

### Value Proposition

- **Correctness**: Proves the algorithm preserves total mass as designed
- **Confidence**: Enables safe use of staggered stepping in production
- **Regression Prevention**: Catches future changes that break conservation
- **Documentation**: Tests serve as executable specification

## Scope

### In Scope

- Mass conservation test harness for all theta modes
- Tests with varying particle counts (100, 1000, 10000)
- Kelvin effect stress tests (small particles < 10 nm)
- Supersaturation and subsaturation scenarios
- Tolerance specification (1e-12 relative error)

### Out of Scope

- Core staggered stepping logic (E1-F1, E1-F2)
- Builder and factory integration (E1-F3)
- Performance benchmarks (E1-F5)
- Documentation and examples (E1-F6)

## Dependencies

### Upstream

- **E1-F1**: Core Staggered Stepping Logic (must be complete)
- **E1-F2**: Batch-Wise Stepping Mode (must be complete)
- **E1-F3**: Builder and Factory Integration (recommended for ergonomic test
  setup)

### Downstream

- E1-F5: Stability and Performance Benchmarks (may extend these tests)
- E1-F6: Documentation and Examples (references these tests)

## Phase Checklist

- [ ] **E1-F4-P1:** Create mass conservation test harness with comprehensive
      tests
  - Issue: TBD | Size: S | Status: Not Started
  - File: `particula/dynamics/condensation/tests/staggered_mass_conservation_test.py`
  - Test all three modes: `"half"`, `"random"`, `"batch"`
  - Verify `|total_initial - total_final| < tolerance` (1e-12 relative)
  - Test with varying particle counts (100, 1000, 10000)
  - Include parametrized tests for mode and particle count combinations

- [ ] **E1-F4-P2:** Add Kelvin effect stress tests
  - Issue: TBD | Size: S | Status: Not Started
  - Test small particles (< 10 nm) with high Kelvin curvature
  - Verify evaporation occurs correctly under vapor competition
  - Test supersaturation and subsaturation scenarios
  - Verify conservation holds even with evaporation/condensation switching

## Critical Testing Requirements

- **No Coverage Modifications**: Keep coverage thresholds at 80% or higher.
- **Self-Contained Tests**: Test file should be runnable independently.
- **Test-First Completion**: Tests define success criteria for the algorithm.
- **Tolerance Documentation**: Clearly document the 1e-12 relative tolerance.

## Testing Strategy

### Mass Conservation Tests

Location: `particula/dynamics/condensation/tests/staggered_mass_conservation_test.py`

**Test Cases:**

- [ ] `test_mass_conservation_half_mode_100_particles`
- [ ] `test_mass_conservation_half_mode_1000_particles`
- [ ] `test_mass_conservation_half_mode_10000_particles`
- [ ] `test_mass_conservation_random_mode_100_particles`
- [ ] `test_mass_conservation_random_mode_1000_particles`
- [ ] `test_mass_conservation_random_mode_10000_particles`
- [ ] `test_mass_conservation_batch_mode_100_particles`
- [ ] `test_mass_conservation_batch_mode_1000_particles`
- [ ] `test_mass_conservation_batch_mode_10000_particles`
- [ ] `test_mass_conservation_multi_step` (10, 100, 1000 steps)
- [ ] `test_mass_conservation_large_time_step` (1s, 10s, 100s)

### Kelvin Effect Tests

**Test Cases:**

- [ ] `test_kelvin_small_particles_evaporation`
- [ ] `test_kelvin_mass_conservation_during_evaporation`
- [ ] `test_supersaturation_condensation`
- [ ] `test_subsaturation_evaporation`
- [ ] `test_mixed_supersaturation_subsaturation`
- [ ] `test_kelvin_critical_diameter_behavior`

## Technical Approach

### Test Harness Design

```python
"""Mass conservation tests for staggered condensation."""

import numpy as np
import pytest
from numpy.typing import NDArray

from particula.dynamics.condensation import CondensationIsothermalStaggered
from particula.gas import AtmosphereBuilder, GasSpeciesBuilder
from particula.particles import (
    ParticleRepresentationBuilder,
    ParticleResolvedSpeciatedMassBuilder,
)


def calculate_total_mass(
    particle,
    gas_species,
) -> float:
    """Calculate total mass in system (particles + gas).
    
    Args:
        particle: ParticleRepresentation with speciated mass.
        gas_species: GasSpecies with concentration.
        
    Returns:
        Total mass in kg (particle mass + gas mass).
    """
    # Sum all particle mass (across all species)
    particle_mass = np.sum(particle.get_mass())
    
    # Gas mass = concentration (kg/m³) summed across species
    # Note: concentration is already in mass units for our setup
    gas_mass = np.sum(gas_species.get_concentration())
    
    return float(particle_mass + gas_mass)


@pytest.fixture
def create_test_system():
    """Factory fixture to create particle systems with specified particle count.
    
    Returns:
        Function that creates (particle, gas_species) tuple.
    """
    def _create(n_particles: int):
        """Create particle system with n_particles particles.
        
        Args:
            n_particles: Number of particles to create.
            
        Returns:
            Tuple of (particle, gas_species).
        """
        # Create gas species (water vapor)
        gas_species = (
            GasSpeciesBuilder()
            .set_molar_mass(0.018, "kg/mol")
            .set_vapor_pressure(3000.0, "Pa")  # ~saturated at 298 K
            .set_concentration(0.01, "kg/m^3")  # initial gas concentration
            .build()
        )
        
        # Create particle distribution
        # Log-normal distribution of particle masses
        rng = np.random.default_rng(42)  # reproducible
        particle_masses = rng.lognormal(
            mean=-20, sigma=1.0, size=n_particles
        )  # kg
        
        # Use ParticleResolvedSpeciatedMass distribution
        distribution = (
            ParticleResolvedSpeciatedMassBuilder()
            .set_mass(particle_masses, "kg")
            .set_density(1000.0, "kg/m^3")
            .build()
        )
        
        particle = (
            ParticleRepresentationBuilder()
            .set_distribution(distribution)
            .set_concentration(np.ones(n_particles), "1/m^3")
            .build()
        )
        
        return particle, gas_species
    
    return _create


class TestMassConservation:
    """Test suite for mass conservation in staggered condensation."""

    RELATIVE_TOLERANCE = 1e-12

    @pytest.mark.parametrize("theta_mode", ["half", "random", "batch"])
    @pytest.mark.parametrize("n_particles", [100, 1000, 10000])
    def test_mass_conservation(
        self,
        theta_mode: str,
        n_particles: int,
        create_test_system,
    ):
        """Verify total mass is conserved to within tolerance."""
        particle, gas_species = create_test_system(n_particles)

        initial_mass = calculate_total_mass(particle, gas_species)

        strategy = CondensationIsothermalStaggered(
            molar_mass=0.018,  # kg/mol for water
            theta_mode=theta_mode,
            num_batches=10,
            random_state=42,  # reproducibility
        )

        particle_out, gas_out = strategy.step(
            particle=particle,
            gas_species=gas_species,
            temperature=298.0,
            pressure=101325.0,
            time_step=1.0,
        )

        final_mass = calculate_total_mass(particle_out, gas_out)

        relative_error = abs(final_mass - initial_mass) / initial_mass
        assert relative_error < self.RELATIVE_TOLERANCE, (
            f"Mass conservation violated for theta_mode={theta_mode}, "
            f"n_particles={n_particles}: relative error = {relative_error:.2e}"
        )
```

### Kelvin Effect Test Design

```python
class TestKelvinEffectConservation:
    """Test mass conservation under Kelvin effect conditions."""

    @pytest.fixture
    def small_particles(self):
        """Create particles with diameter < 10 nm for high Kelvin curvature."""
        diameters = np.array([5e-9, 7e-9, 9e-9])  # 5, 7, 9 nm
        # ... create particles
        return particle, gas_species, air_volume

    def test_kelvin_evaporation_conserves_mass(self, small_particles):
        """Verify mass conservation when small particles evaporate."""
        particle, gas_species, air_volume = small_particles

        # Set subsaturation to trigger evaporation
        gas_species.concentration *= 0.8  # 80% saturation

        initial_mass = calculate_total_mass(particle, gas_species, air_volume)

        strategy = CondensationIsothermalStaggered(
            molar_mass=gas_species.molar_mass,
            theta_mode="random",
        )

        # Step should cause evaporation due to Kelvin effect
        particle, gas_species = strategy.step(...)

        final_mass = calculate_total_mass(particle, gas_species, air_volume)

        assert abs(final_mass - initial_mass) / initial_mass < 1e-12
```

### Tolerance Specification

| Scenario | Relative Tolerance | Rationale |
|----------|-------------------|-----------|
| Single step | 1e-12 | Machine precision for double |
| Multi-step (1000) | 1e-10 | Accumulated error allowance |
| Kelvin effect | 1e-12 | Same precision regardless of physics |
| Large time step | 1e-12 | Algorithm design, not numerical issue |

## Success Criteria

- [ ] Mass conservation verified for all theta modes (half, random, batch)
- [ ] Conservation holds for particle counts: 100, 1000, 10000
- [ ] Kelvin effect tests pass with evaporation scenarios
- [ ] Supersaturation and subsaturation scenarios conserve mass
- [ ] Multi-step simulations maintain cumulative tolerance
- [ ] All tests pass with documented tolerances
- [ ] Tests are parametrized and maintainable

## Usage Example

```python
# Running mass conservation tests
pytest particula/dynamics/condensation/tests/staggered_mass_conservation_test.py -v

# Running with specific parameters
pytest -k "test_mass_conservation and random and 1000"

# Running Kelvin effect tests
pytest -k "kelvin"
```

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| 10000 particle tests too slow for CI | Medium | Low | Mark with `@pytest.mark.slow` if > 5s |
| Fixture setup complexity | Low | Medium | Use factory fixture pattern for flexibility |
| Tolerance too strict for numerical precision | Low | High | Document rationale; relax to 1e-10 if needed |

## Change Log

| Date       | Change                                | Author       |
|------------|---------------------------------------|--------------|
| 2025-12-23 | Initial feature documentation created | ADW Workflow |
| 2025-12-29 | Added complete fixture code, factory pattern | ADW Workflow |

[epic-e1]: ../epics/E1-staggered-condensation-stepping.md
