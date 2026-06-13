# Feature E1-F1: Core Staggered Stepping Logic

**Status:** Not Started
**Priority:** P2
**Assignees:** TBD
**Labels:** feature, dynamics, condensation, staggered-stepping
**Milestone:** v0.3.x
**Size:** M (~200 LOC core + tests)

**Start Date:** TBD
**Target Date:** TBD
**Created:** 2025-12-23
**Updated:** 2025-12-23

**Parent Epic:** [E1: Staggered ODE Stepping][epic-e1]
**Related Issues:** TBD (create during implementation)
**Related PRs:** TBD

---

## Overview

Implement the core `CondensationIsothermalStaggered` strategy class that provides
staggered ODE stepping for particle-resolved condensation. This feature
introduces the fundamental two-pass stepping algorithm with configurable theta
modes (`"half"`, `"random"`, `"batch"`) to improve numerical stability and mass
conservation.

### Problem Statement

The current `CondensationIsothermal.step()` updates all particles simultaneously
against a fixed gas concentration. At large time steps, this causes:

- Numerical instability from competing vapor uptake
- Unrealistic mass transfer when multiple particles deplete the same vapor
- Accumulated error in total mass conservation

### Value Proposition

- **Mass Conservation**: Two-pass algorithm preserves total (gas + particle) mass
- **Stability**: Staggered updates reduce numerical noise from simultaneous
  growth
- **Flexibility**: Three theta modes support different use cases (deterministic
  vs stochastic vs batch)
- **API Compatibility**: Preserves existing `step()` signature for drop-in
  replacement

## Scope

### In Scope

- `CondensationIsothermalStaggered` class extending `CondensationStrategy`
- Three theta modes: `"half"`, `"random"`, `"batch"`
- `_get_theta_values()` helper method for theta array generation
- `_make_batches()` helper method for particle index batching
- Two-pass `step()` implementation with cumulative gas updates
- Unit tests for all new methods and modes

### Out of Scope

- Builder and factory integration (E1-F3)
- Batch-wise Gauss-Seidel stepping details (E1-F2)
- Mass conservation validation tests (E1-F4)
- Performance benchmarks (E1-F5)
- Documentation and examples (E1-F6)

## Dependencies

### Upstream

- None blocking. Condensation module is stable.
- Requires `ParticleResolvedSpeciatedMass` distribution strategy (exists)

### Downstream

- E1-F2: Batch-Wise Stepping Mode (extends batch logic)
- E1-F3: Builder and Factory Integration (wraps this class)
- E1-F4: Mass Conservation Validation (tests this class)

## Phase Checklist

- [ ] **E1-F1-P1:** Create `CondensationIsothermalStaggered` class skeleton with
      tests
  - Issue: TBD | Size: S | Status: Not Started
  - File: `particula/dynamics/condensation/condensation_strategies.py`
  - Add new class inheriting from `CondensationStrategy`
  - Add `theta_mode` parameter: `"half"`, `"random"`, `"batch"`
  - Add `num_batches` parameter (default: 1)
  - Add `shuffle_each_step` parameter (default: True for random mode)
  - Add `random_state` parameter for reproducible random sequences (optional)
  - Inherit all parent parameters: `molar_mass`, `diffusion_coefficient`,
    `accommodation_coefficient`, `update_gases`, `skip_partitioning_indices`
  - Include unit tests for class instantiation and parameter validation
  - Test file: `particula/dynamics/condensation/tests/condensation_strategies_test.py`

- [ ] **E1-F1-P2:** Implement `_get_theta_values()` helper method with tests
  - Issue: TBD | Size: S | Status: Not Started
  - Generate theta array based on mode:
    - `"half"`: `np.full(n_particles, 0.5)`
    - `"random"`: `np.random.default_rng(self.random_state).uniform(0, 1, n_particles)`
    - `"batch"`: `np.ones(n_particles)` (batching handled separately)
  - Handle single-particle case (n_particles=1) gracefully
  - Include unit tests for each mode verifying shape and value ranges
  - Test reproducibility when `random_state` is set

- [ ] **E1-F1-P3:** Implement `_make_batches()` helper method with tests
  - Issue: TBD | Size: S | Status: Not Started
  - Shuffle particle indices if `shuffle_each_step=True`
  - Use `np.random.default_rng(self.random_state)` for reproducible shuffling
  - Divide indices into `num_batches` groups using `np.array_split()`
  - Return list of index arrays (NDArray[np.intp])
  - Handle edge case: `num_batches > n_particles` → clip to n_particles
  - Handle edge case: `num_batches = 1` → single batch with all indices
  - Include unit tests for batch creation, shuffling, and edge cases

- [ ] **E1-F1-P4:** Implement two-pass staggered `step()` method with tests
  - Issue: TBD | Size: M | Status: Not Started
  - **Algorithm Details:**
    1. Get particle count from `particle.get_concentration().shape[0]`
    2. Generate theta values via `_get_theta_values(n_particles)`
    3. Create batches via `_make_batches(n_particles)`
    4. Copy gas concentration: `gas_conc = gas_species.get_concentration().copy()`
    5. **Pass 1**: For each batch, for each particle index:
       - Compute `dt_local = theta[idx] * time_step`
       - Call parent `mass_transfer_rate()` for single particle
       - Accumulate mass change and update gas concentration
    6. **Pass 2**: Same as Pass 1 with `dt_local = (1 - theta[idx]) * time_step`
    7. Apply mass changes to particle via `particle.add_mass()`
    8. Update gas species if `self.update_gases` is True
  - Preserve existing API: `step(particle, gas_species, temperature, pressure,
    time_step)`
  - Handle speciated mass: work with per-species mass arrays, not just total
  - Include integration tests verifying basic mass conservation
  - Ensure compatibility with `ParticleResolvedSpeciatedMass` distribution

## Critical Testing Requirements

- **No Coverage Modifications**: Keep coverage thresholds at 80% or higher.
- **Self-Contained Tests**: Ship `*_test.py` suites that prove the feature works
  end-to-end.
- **Test-First Completion**: Tests must exist and pass before finishing each
  phase.
- **80%+ Coverage**: Maintain at least 80% coverage for touched code.

## Testing Strategy

### Unit Tests

Location: `particula/dynamics/condensation/tests/condensation_strategies_test.py`

**Test Cases:**

- [ ] Class instantiation with valid parameters succeeds
- [ ] Invalid `theta_mode` raises ValueError
- [ ] Invalid `num_batches` (< 1) raises ValueError
- [ ] `_get_theta_values("half")` returns array of 0.5 values
- [ ] `_get_theta_values("random")` returns values in [0, 1]
- [ ] `_get_theta_values("batch")` returns array of 1.0 values
- [ ] `_make_batches()` divides indices into correct number of batches
- [ ] `_make_batches()` shuffles indices when `shuffle_each_step=True`
- [ ] `_make_batches()` preserves indices when `shuffle_each_step=False`

### Integration Tests

Location: `particula/dynamics/condensation/tests/condensation_strategies_test.py`

**Test Cases:**

- [ ] `step()` with `theta_mode="half"` produces valid output
- [ ] `step()` with `theta_mode="random"` produces valid output
- [ ] `step()` API matches existing `CondensationIsothermal.step()` signature
- [ ] Output particle masses are non-negative
- [ ] Gas concentration remains non-negative

## Technical Approach

### Architecture

```
CondensationStrategy (ABC)
    └── CondensationIsothermal (existing)
    └── CondensationIsothermalStaggered (new)
            ├── theta_mode: str
            ├── num_batches: int
            ├── shuffle_each_step: bool
            ├── _get_theta_values() -> NDArray
            ├── _make_batches() -> list[NDArray]
            └── step() -> tuple[Particle, GasSpecies]
```

### Algorithm (Detailed)

```python
def step(self, particle, gas_species, temperature, pressure, time_step):
    """Two-pass staggered stepping algorithm."""
    n_particles = particle.get_concentration().shape[0]
    theta = self._get_theta_values(n_particles)
    batches = self._make_batches(n_particles)

    # Working copy of gas concentration (per-species array)
    gas_concentration = gas_species.get_concentration().copy()
    
    # Track cumulative mass changes per particle (shape: n_particles x n_species)
    mass_changes = np.zeros_like(particle.get_species_mass())

    # Pass 1: fractional step with theta
    for batch in batches:
        batch_dm_total = np.zeros(gas_concentration.shape)  # per-species
        for idx in batch:
            dt_local = theta[idx] * time_step
            # Calculate mass transfer for this particle at current gas state
            dm = self._calculate_single_particle_transfer(
                particle, idx, gas_concentration, temperature, pressure, dt_local
            )
            mass_changes[idx] += dm
            batch_dm_total += dm
        # Gauss-Seidel: update gas after each batch
        gas_concentration -= batch_dm_total

    # Pass 2: remaining step with (1 - theta)
    for batch in batches:
        batch_dm_total = np.zeros(gas_concentration.shape)
        for idx in batch:
            dt_local = (1 - theta[idx]) * time_step
            dm = self._calculate_single_particle_transfer(
                particle, idx, gas_concentration, temperature, pressure, dt_local
            )
            mass_changes[idx] += dm
            batch_dm_total += dm
        gas_concentration -= batch_dm_total

    # Apply accumulated mass changes
    particle.add_mass(added_mass=mass_changes)
    
    if self.update_gases:
        gas_species.add_concentration(
            added_concentration=-(mass_changes.sum(axis=0))
        )
    
    return particle, gas_species


def _calculate_single_particle_transfer(
    self, particle, idx, gas_concentration, temperature, pressure, dt_local
):
    """Calculate mass transfer for a single particle.
    
    This is a helper that extracts single-particle data and calls
    the existing mass_transfer_rate calculation, then scales by dt_local.
    """
    # Extract single particle data (implementation detail)
    # Use existing mass_transfer_rate() logic but for one particle
    # Return: dm array of shape (n_species,)
    pass
```

**Key Implementation Notes:**

1. **Per-Species Tracking**: Mass changes are tracked per-species, not just total
   mass, to support multi-component aerosols.

2. **Gauss-Seidel Updates**: Gas concentration is updated after each batch, not
   just at the end of each pass. This provides better convergence properties.

3. **Compatibility**: The `_calculate_single_particle_transfer()` helper must work
   with the existing `mass_transfer_rate()` infrastructure from the parent class.

4. **Memory Efficiency**: Mass changes are accumulated in a working array and
   applied once at the end, avoiding multiple particle state mutations.

### API Surface

```python
from particula.dynamics.condensation import CondensationIsothermalStaggered

# Full constructor with all parameters
strategy = CondensationIsothermalStaggered(
    # Required (inherited from CondensationStrategy)
    molar_mass=0.018,  # kg/mol (water)
    # Optional inherited parameters
    diffusion_coefficient=2e-5,  # m^2/s (default)
    accommodation_coefficient=1.0,  # unitless (default)
    update_gases=True,  # update gas concentrations (default)
    skip_partitioning_indices=None,  # species to skip (default)
    # New staggered-specific parameters
    theta_mode="random",  # or "half", "batch"
    num_batches=1,  # number of Gauss-Seidel batches (default)
    shuffle_each_step=True,  # randomize order each step (default)
    random_state=None,  # int or None for reproducibility (default)
)

# Step method (same signature as parent)
particle, gas = strategy.step(
    particle=particle,
    gas_species=gas_species,
    temperature=298.0,  # K
    pressure=101325.0,  # Pa
    time_step=1.0,  # seconds
)

# Other inherited methods still work
rate_array = strategy.rate(particle, gas_species, 298.0, 101325.0)
mass_rate = strategy.mass_transfer_rate(particle, gas_species, 298.0, 101325.0)
```

## Success Criteria

- [ ] `CondensationIsothermalStaggered` class instantiates with all theta modes
- [ ] `_get_theta_values()` produces correct arrays for each mode
- [ ] `_make_batches()` correctly divides and optionally shuffles indices
- [ ] `step()` executes two-pass algorithm and returns valid particle/gas
- [ ] All unit and integration tests pass
- [ ] Code coverage >= 80% for new code
- [ ] Code review approved

## Usage Example

```python
import numpy as np
from particula.dynamics.condensation import CondensationIsothermalStaggered

# Create staggered condensation strategy with reproducible random state
strategy = CondensationIsothermalStaggered(
    molar_mass=0.018,  # kg/mol for water
    theta_mode="random",
    num_batches=1,
    shuffle_each_step=True,
    random_state=42,  # for reproducibility in tests
)

# Step the condensation process
particle_updated, gas_updated = strategy.step(
    particle=particle,
    gas_species=gas_species,
    temperature=298.0,
    pressure=101325.0,
    time_step=0.1,
)

# Verify mass conservation
initial_total_mass = particle.get_mass().sum() + gas_species.get_concentration().sum()
final_total_mass = particle_updated.get_mass().sum() + gas_updated.get_concentration().sum()
assert np.isclose(initial_total_mass, final_total_mass, rtol=1e-12)
```

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Performance regression from per-particle loops | Medium | Medium | Profile and optimize with NumPy vectorization where possible; document expected overhead |
| Incompatibility with non-particle-resolved distributions | Low | High | Explicitly check distribution type in `step()` and raise clear error |
| Random state affecting reproducibility | Low | Medium | Add `random_state` parameter; document in tests |
| Mass conservation violations at edge cases | Medium | High | Extensive E1-F4 test coverage; use 1e-12 relative tolerance |

## Change Log

| Date       | Change                                | Author       |
|------------|---------------------------------------|--------------|
| 2025-12-23 | Initial feature documentation created | ADW Workflow |
| 2025-12-29 | Expanded algorithm details, added random_state parameter, risks section | ADW Workflow |

[epic-e1]: ../epics/E1-staggered-condensation-stepping.md
