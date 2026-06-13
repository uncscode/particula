# Feature E1-F5: Stability and Performance Benchmarks

**Status:** In Progress
**Priority:** P3
**Assignees:** ADW Workflow
**Labels:** feature, dynamics, condensation, benchmarks, performance
**Milestone:** v0.3.x
**Size:** S (~100 LOC tests)

**Start Date:** 2026-01-02
**Target Date:** TBD
**Created:** 2025-12-23
**Updated:** 2026-01-02

**Parent Epic:** [E1: Staggered ODE Stepping][epic-e1]
**Related Issues:** #99 (parent), #137 (E1-F5-P2 performance benchmarks)
**Related PRs:** TBD

---

## Overview

Create benchmark test suites that measure stability improvements and performance
characteristics of staggered condensation stepping compared to simultaneous
updates. These benchmarks document the tradeoffs between accuracy, stability,
and runtime.

### Problem Statement

While mass conservation tests (E1-F4) verify correctness, users need to
understand:

- How much larger time steps can be used before instability?
- What is the runtime overhead of staggered stepping?
- How does performance scale with particle count?
- Which theta mode offers the best stability/performance tradeoff?

### Value Proposition

- **Guidance**: Helps users choose appropriate theta mode and batch size
- **Documentation**: Quantifies stability improvements claimed in epic
- **Regression Detection**: Catches performance regressions in future changes
- **Validation**: Proves 10x time step improvement and <2x runtime targets

## Scope

### In Scope

- Stability benchmark comparing variance vs simultaneous update
- Time step limit tests (1s, 10s, 100s)
- Performance benchmarks (1k, 10k, 100k particles)
- Runtime comparison (staggered vs simultaneous)
- `@pytest.mark.slow` marking for CI

### Out of Scope

- Core staggered stepping logic (E1-F1, E1-F2)
- Builder and factory integration (E1-F3)
- Mass conservation validation (E1-F4)
- Documentation and examples (E1-F6)
- JAX/GPU acceleration benchmarks (future epic)

## Dependencies

### Upstream

- **E1-F1**: Core Staggered Stepping Logic (must be complete)
- **E1-F2**: Batch-Wise Stepping Mode (must be complete)
- **E1-F3**: Builder and Factory Integration (recommended)
- **E1-F4**: Mass Conservation Validation (should pass first)

### Downstream

- E1-F6: Documentation and Examples (references benchmark results)

## Phase Checklist

- [ ] **E1-F5-P1:** Create stability benchmark tests
  - Issue: TBD | Size: S | Status: Not Started
  - File: `particula/dynamics/condensation/tests/staggered_stability_test.py`
  - Compare variance in particle size distribution vs simultaneous update
  - Test large time steps (1s, 10s, 100s) where simultaneous fails
  - Document stability improvement metrics
  - Mark appropriate tests with `@pytest.mark.slow`

- [x] **E1-F5-P2:** Create performance benchmark tests
  - Issue: #137 | Size: S | Status: Completed (slow+performance suite added)
  - File: `particula/dynamics/condensation/tests/staggered_performance_test.py`
  - Run: `pytest particula/dynamics/condensation/tests/staggered_performance_test.py -v -m "slow and performance"` (CI-excluded slow+performance)
  - Coverage: overhead target <2x vs simultaneous; scaling at 1k/10k/100k; theta modes half/random/batch with deterministic seeds
  - Notes: module-level slow+performance markers; batch clipping/logging, nan-safe pressure handling, deterministic shuffling; mass transfer inventory guard; compile guard for imports

## Critical Testing Requirements

- **Slow Test Marking**: All benchmarks must use `@pytest.mark.slow`
- **Performance Test Marking**: Use `@pytest.mark.performance` for timing benchmarks
- **Reproducibility**: Set random seeds for deterministic results
- **Documentation**: Record benchmark results in test docstrings
- **CI Exclusion**: Slow and performance tests excluded from normal CI runs
  - Run with: `pytest particula/dynamics/condensation/tests/staggered_performance_test.py -v -m "slow and performance"`
  - Skip with: `pytest -m "not slow and not performance"`

## Testing Strategy

### Stability Benchmarks

Location: `particula/dynamics/condensation/tests/staggered_stability_test.py`

**Test Cases:**

- [ ] `test_stability_variance_comparison` - Compare variance vs simultaneous
- [ ] `test_stability_large_time_step_1s` - Verify stability at dt=1s
- [ ] `test_stability_large_time_step_10s` - Verify stability at dt=10s
- [ ] `test_stability_large_time_step_100s` - Verify stability at dt=100s
- [ ] `test_stability_mode_comparison` - Compare half vs random vs batch
- [ ] `test_stability_batch_count_effect` - Effect of num_batches on stability

### Performance Benchmarks

Location: `particula/dynamics/condensation/tests/staggered_performance_test.py`

**Test Cases:**

- [x] `test_performance_1k_particles` - Runtime with 1,000 particles (<2x overhead target)
- [x] `test_performance_10k_particles` - Runtime with 10,000 particles (<2x overhead target)
- [x] `test_performance_100k_particles` - Runtime with 100,000 particles (capped iterations, <2x target)
- [x] `test_performance_scaling` - Verify O(n) scaling across 1k/10k/100k
- [x] `test_performance_mode_comparison` - Runtime: half vs random vs batch with deterministic seeds
- [x] `test_performance_vs_simultaneous` - Overhead vs baseline (module-level slow+performance markers)

**Results & Targets (P2):**
- Overhead target: <2x staggered vs simultaneous across 1k/10k/100k (enforced in benchmarks).
- Scaling expectation: roughly O(n) wall time across 1k/10k/100k cases with capped iterations.
- Theta mode comparison: half, random, batch timed with deterministic seeds; printed timings for documentation.
- Execution: `pytest particula/dynamics/condensation/tests/staggered_performance_test.py -v -m "slow and performance"` (slow+performance markers keep this out of CI).

## Technical Approach

### Stability Benchmark Design

```python
import numpy as np
import pytest
from particula.dynamics.condensation import (
    CondensationIsothermal,
    CondensationIsothermalStaggered,
)


@pytest.mark.slow
class TestStabilityBenchmarks:
    """Stability benchmarks for staggered condensation."""

    def calculate_size_variance(self, particle):
        """Calculate variance in particle size distribution."""
        diameters = particle.get_diameter()
        return np.var(diameters)

    @pytest.mark.parametrize("time_step", [1.0, 10.0, 100.0])
    def test_stability_large_time_step(self, time_step):
        """Test stability at large time steps.

        Expected: Staggered stepping remains stable where simultaneous fails.
        Target: 10x larger time steps than simultaneous.
        """
        particle, gas_species = self.setup_system(n_particles=1000)

        # Simultaneous stepping
        simultaneous = CondensationIsothermal(molar_mass=0.018)
        try:
            p_sim, g_sim = simultaneous.step(
                particle.copy(), gas_species.copy(), 298.0, 101325.0, time_step
            )
            simultaneous_stable = self.is_stable(p_sim)
        except (ValueError, RuntimeWarning):
            simultaneous_stable = False

        # Staggered stepping
        staggered = CondensationIsothermalStaggered(
            molar_mass=0.018,
            theta_mode="random",
            num_batches=10,
        )
        p_stag, g_stag = staggered.step(
            particle.copy(), gas_species.copy(), 298.0, 101325.0, time_step
        )
        staggered_stable = self.is_stable(p_stag)

        # Document result
        print(f"dt={time_step}s: simultaneous={simultaneous_stable}, "
              f"staggered={staggered_stable}")

        # Staggered should be stable at larger time steps
        if time_step >= 10.0:
            assert staggered_stable, "Staggered should be stable at dt=10s+"

    def is_stable(self, particle):
        """Check if particle distribution is numerically stable."""
        masses = particle.mass
        return (
            np.all(np.isfinite(masses)) and
            np.all(masses >= 0) and
            np.var(masses) < 1e10  # No explosion
        )
```

### Performance Benchmark Design

```python
import time
import pytest


@pytest.mark.slow
class TestPerformanceBenchmarks:
    """Performance benchmarks for staggered condensation."""

    @pytest.mark.parametrize("n_particles", [1000, 10000, 100000])
    def test_performance_scaling(self, n_particles):
        """Measure runtime scaling with particle count.

        Target: < 2x overhead vs simultaneous stepping.
        """
        particle, gas_species = self.setup_system(n_particles)

        # Benchmark simultaneous
        simultaneous = CondensationIsothermal(molar_mass=0.018)
        start = time.perf_counter()
        for _ in range(10):
            simultaneous.step(
                particle.copy(), gas_species.copy(), 298.0, 101325.0, 0.1
            )
        simultaneous_time = (time.perf_counter() - start) / 10

        # Benchmark staggered
        staggered = CondensationIsothermalStaggered(
            molar_mass=0.018,
            theta_mode="random",
            num_batches=10,
        )
        start = time.perf_counter()
        for _ in range(10):
            staggered.step(
                particle.copy(), gas_species.copy(), 298.0, 101325.0, 0.1
            )
        staggered_time = (time.perf_counter() - start) / 10

        overhead = staggered_time / simultaneous_time

        print(f"n={n_particles}: simultaneous={simultaneous_time:.4f}s, "
              f"staggered={staggered_time:.4f}s, overhead={overhead:.2f}x")

        # Target: < 2x overhead
        assert overhead < 2.0, f"Overhead {overhead:.2f}x exceeds 2x target"

    @pytest.mark.parametrize("theta_mode", ["half", "random", "batch"])
    def test_performance_mode_comparison(self, theta_mode):
        """Compare runtime across theta modes."""
        particle, gas_species = self.setup_system(n_particles=10000)

        strategy = CondensationIsothermalStaggered(
            molar_mass=0.018,
            theta_mode=theta_mode,
            num_batches=10,
        )

        start = time.perf_counter()
        for _ in range(10):
            strategy.step(
                particle.copy(), gas_species.copy(), 298.0, 101325.0, 0.1
            )
        elapsed = (time.perf_counter() - start) / 10

        print(f"theta_mode={theta_mode}: {elapsed:.4f}s per step")
```

### Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Stability improvement | 10x larger dt | Max stable dt comparison |
| Runtime overhead | < 2x | Staggered / Simultaneous time |
| Scaling | O(n) | Runtime vs particle count |
| Memory | < 1.5x | Peak memory comparison |

## Success Criteria

- [ ] Stability benchmarks demonstrate 10x time step improvement
- [ ] Performance benchmarks show < 2x runtime overhead
- [ ] Scaling tests verify O(n) behavior
- [ ] All benchmarks marked with `@pytest.mark.slow`
- [ ] Results documented in test docstrings
- [ ] Benchmark results reproducible (random seeds set)

## Usage Example

```python
# Run stability benchmarks
pytest particula/dynamics/condensation/tests/staggered_stability_test.py -v -m slow

# Run performance benchmarks (slow+performance; excluded from CI)
pytest particula/dynamics/condensation/tests/staggered_performance_test.py -v -m "slow and performance"

# Run all slow tests
pytest -m slow -v

# Skip slow/performance tests in normal CI
pytest -m "not slow and not performance"
```

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Benchmarks flaky due to system load | High | Medium | Run multiple iterations; report mean ± std |
| 100k particle tests use too much memory | Medium | Medium | Monitor memory; reduce if needed |
| Performance targets not met | Medium | High | Profile and optimize in follow-up PR |

## Change Log

| Date       | Change                                | Author       |
|------------|---------------------------------------|--------------|
| 2025-12-23 | Initial feature documentation created | ADW Workflow |
| 2025-12-29 | Added @pytest.mark.performance, risks section | ADW Workflow |
| 2026-01-02 | Updated P2 performance benchmarks, markers, and run command (#137) | ADW Workflow |

[epic-e1]: ../epics/E1-staggered-condensation-stepping.md
