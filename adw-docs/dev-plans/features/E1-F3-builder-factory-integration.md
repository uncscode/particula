# Feature E1-F3: Builder and Factory Integration

**Status:** Not Started
**Priority:** P2
**Assignees:** TBD
**Labels:** feature, dynamics, condensation, builders, factory
**Milestone:** v0.3.x
**Size:** M (~150 LOC core + tests)

**Start Date:** TBD
**Target Date:** TBD
**Created:** 2025-12-23
**Updated:** 2025-12-23

**Parent Epic:** [E1: Staggered ODE Stepping][epic-e1]
**Related Issues:** TBD
**Related PRs:** TBD

---

## Overview

Add builder and factory support for the `CondensationIsothermalStaggered`
strategy, enabling ergonomic configuration through fluent setters and factory
instantiation. This aligns staggered condensation with the established
builder/factory patterns used throughout particula.

### Problem Statement

While the core staggered stepping class (E1-F1, E1-F2) provides the algorithm,
users need:

- Fluent builder API for configuring staggered parameters
- Factory integration for consistent strategy instantiation
- Public exports via `particula.dynamics` namespace

### Value Proposition

- **Ergonomics**: Fluent builder API with method chaining
- **Consistency**: Matches patterns used in wall loss, coagulation, etc.
- **Discoverability**: Factory exposes strategy via `strategy_type` string
- **Validation**: Builder enforces valid parameter combinations

## Scope

### In Scope

- `CondensationIsothermalStaggeredBuilder` class with fluent setters
- Factory registration under `"isothermal_staggered"` strategy type
- Public exports via `particula.dynamics` and `particula.dynamics.condensation`
- Unit tests for builder, factory, and exports

### Out of Scope

- Core staggered stepping logic (E1-F1, E1-F2)
- Mass conservation validation (E1-F4)
- Performance benchmarks (E1-F5)
- Documentation and examples (E1-F6)

## Dependencies

### Upstream

- **E1-F1**: Core Staggered Stepping Logic (must be complete)
- **E1-F2**: Batch-Wise Stepping Mode (must be complete)
- Existing `BuilderABC` and `StrategyFactoryABC` patterns

### Downstream

- E1-F4: Mass Conservation Validation (may use factory)
- E1-F5: Stability and Performance Benchmarks (may use factory)
- E1-F6: Documentation and Examples (documents factory usage)

## Phase Checklist

- [ ] **E1-F3-P1:** Create `CondensationIsothermalStaggeredBuilder` with tests
  - Issue: TBD | Size: S | Status: Not Started
  - File: `particula/dynamics/condensation/condensation_builders.py`
  - Methods: `set_theta_mode()`, `set_num_batches()`, `set_shuffle_each_step()`
  - Inherit common condensation methods from existing mixin
  - Include validation for theta_mode values and num_batches >= 1
  - Include builder tests for all setters and validation

- [ ] **E1-F3-P2:** Register in `CondensationFactory` with tests
  - Issue: TBD | Size: S | Status: Not Started
  - File: `particula/dynamics/condensation/condensation_factories.py`
  - Add `"isothermal_staggered"` strategy type mapping
  - Wire up factory parameters to builder
  - Include factory tests verifying strategy creation

- [ ] **E1-F3-P3:** Export from `particula.dynamics` namespace with tests
  - Issue: TBD | Size: XS | Status: Not Started
  - Update `particula/dynamics/__init__.py`
  - Update `particula/dynamics/condensation/__init__.py`
  - Include import smoke tests verifying public API access

## Critical Testing Requirements

- **No Coverage Modifications**: Keep coverage thresholds at 80% or higher.
- **Self-Contained Tests**: Ship `*_test.py` suites that prove builder/factory
  work correctly.
- **Test-First Completion**: Tests must exist and pass before finishing each
  phase.
- **80%+ Coverage**: Maintain at least 80% coverage for touched code.

## Testing Strategy

### Unit Tests

Location: `particula/dynamics/condensation/tests/condensation_builders_test.py`

**Test Cases:**

- [ ] Builder instantiation succeeds
- [ ] `set_theta_mode("half")` sets mode correctly
- [ ] `set_theta_mode("random")` sets mode correctly
- [ ] `set_theta_mode("batch")` sets mode correctly
- [ ] `set_theta_mode("invalid")` raises ValueError
- [ ] `set_num_batches(10)` sets batches correctly
- [ ] `set_num_batches(0)` raises ValueError
- [ ] `set_shuffle_each_step(True)` sets shuffle correctly
- [ ] Method chaining works: `builder.set_x().set_y().build()`
- [ ] `build()` without required parameters raises error

### Factory Tests

Location: `particula/dynamics/condensation/tests/condensation_factories_test.py`

**Test Cases:**

- [ ] Factory returns staggered strategy via `"isothermal_staggered"`
- [ ] Factory parameters propagate to built strategy
- [ ] Unknown strategy type raises ValueError
- [ ] Factory-built strategy has correct theta_mode, num_batches

### Export Tests

Location: `particula/dynamics/tests/condensation_exports_test.py`

**Test Cases:**

- [ ] `from particula.dynamics import CondensationIsothermalStaggered` works
- [ ] `from particula.dynamics import CondensationIsothermalStaggeredBuilder`
      works
- [ ] `from particula.dynamics.condensation import *` includes new classes

## Technical Approach

### Builder Architecture

```python
class CondensationIsothermalStaggeredBuilder(
    CondensationBuilderMixin,
    BuilderABC,
):
    """Builder for CondensationIsothermalStaggered strategy."""

    def __init__(self):
        super().__init__()
        self._theta_mode: str = "half"  # default
        self._num_batches: int = 1  # default
        self._shuffle_each_step: bool = True  # default

    def set_theta_mode(
        self,
        theta_mode: str,
    ) -> "CondensationIsothermalStaggeredBuilder":
        """Set the theta mode for staggered stepping."""
        valid_modes = {"half", "random", "batch"}
        if theta_mode not in valid_modes:
            raise ValueError(f"theta_mode must be one of {valid_modes}")
        self._theta_mode = theta_mode
        return self

    def set_num_batches(
        self,
        num_batches: int,
    ) -> "CondensationIsothermalStaggeredBuilder":
        """Set the number of batches for batch stepping."""
        if num_batches < 1:
            raise ValueError("num_batches must be >= 1")
        self._num_batches = num_batches
        return self

    def set_shuffle_each_step(
        self,
        shuffle: bool,
    ) -> "CondensationIsothermalStaggeredBuilder":
        """Set whether to shuffle particle order each step."""
        self._shuffle_each_step = shuffle
        return self

    def build(self) -> CondensationIsothermalStaggered:
        """Build the staggered condensation strategy."""
        self.pre_build_check()
        return CondensationIsothermalStaggered(
            molar_mass=self._molar_mass,
            theta_mode=self._theta_mode,
            num_batches=self._num_batches,
            shuffle_each_step=self._shuffle_each_step,
        )
```

### Factory Registration

```python
# In condensation_factories.py
class CondensationFactory(StrategyFactoryABC):
    _builders = {
        "isothermal": CondensationIsothermalBuilder,
        "isothermal_staggered": CondensationIsothermalStaggeredBuilder,
        # ... other strategies
    }
```

### API Surface

```python
# Builder usage
from particula.dynamics.condensation import (
    CondensationIsothermalStaggeredBuilder,
)

strategy = (
    CondensationIsothermalStaggeredBuilder()
    .set_molar_mass(0.018)
    .set_theta_mode("random")
    .set_num_batches(10)
    .set_shuffle_each_step(True)
    .build()
)

# Factory usage
from particula.dynamics.condensation import CondensationFactory

factory = CondensationFactory()
strategy = factory.get_strategy(
    strategy_type="isothermal_staggered",
    parameters={
        "molar_mass": 0.018,
        "theta_mode": "random",
        "num_batches": 10,
        "shuffle_each_step": True,
    },
)

# Public export
from particula.dynamics import CondensationIsothermalStaggered
```

## Success Criteria

- [ ] Builder exposes fluent setters for all staggered parameters
- [ ] Builder validates theta_mode and num_batches
- [ ] Factory returns staggered strategy via `"isothermal_staggered"`
- [ ] Public exports available via `particula.dynamics`
- [ ] All unit, factory, and export tests pass
- [ ] Code coverage >= 80% for new code
- [ ] Code review approved

## Usage Example

```python
import particula as par

# Using builder
builder = par.dynamics.CondensationIsothermalStaggeredBuilder()
strategy = (
    builder
    .set_molar_mass(0.018)
    .set_theta_mode("random")
    .set_num_batches(10)
    .build()
)

# Using factory
from particula.dynamics.condensation import CondensationFactory

factory = CondensationFactory()
strategy = factory.get_strategy(
    strategy_type="isothermal_staggered",
    parameters={
        "molar_mass": 0.018,
        "theta_mode": "batch",
        "num_batches": 5,
    },
)
```

## Change Log

| Date       | Change                                | Author       |
|------------|---------------------------------------|--------------|
| 2025-12-23 | Initial feature documentation created | ADW Workflow |

[epic-e1]: ../epics/E1-staggered-condensation-stepping.md
