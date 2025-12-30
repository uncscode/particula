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
  - **Prerequisite**: E1-F1 and E1-F2 must be complete
  - File: `particula/dynamics/condensation/condensation_builder/condensation_isothermal_staggered_builder.py`
  - **Inherit from** (matching existing pattern):
    - `BuilderABC` (from `particula.abc_builder`)
    - `BuilderMolarMassMixin` (from `particula.builder_mixin`)
    - `BuilderDiffusionCoefficientMixin` (from local mixin)
    - `BuilderAccommodationCoefficientMixin` (from local mixin)
    - `BuilderUpdateGasesMixin` (from local mixin)
  - **New methods to add:**
    - `set_theta_mode(theta_mode: str)` → validates against {"half", "random", "batch"}
    - `set_num_batches(num_batches: int)` → validates >= 1
    - `set_shuffle_each_step(shuffle: bool)` → boolean flag
    - `set_random_state(random_state: Optional[int])` → optional reproducibility seed
  - **Required parameters**: `["molar_mass", "diffusion_coefficient", "accommodation_coefficient"]`
  - **Default values**: `theta_mode="half"`, `num_batches=1`, `shuffle_each_step=True`, `random_state=None`
  - Test file: `particula/dynamics/condensation/condensation_builder/tests/condensation_isothermal_staggered_builder_test.py`

- [ ] **E1-F3-P2:** Register in `CondensationFactory` with tests
  - Issue: TBD | Size: S | Status: Not Started
  - File: `particula/dynamics/condensation/condensation_factories.py`
  - Add `"isothermal_staggered"` to `get_builders()` dict:
    ```python
    return {
        "isothermal": CondensationIsothermalBuilder(),
        "isothermal_staggered": CondensationIsothermalStaggeredBuilder(),
    }
    ```
  - Import new builder at top of file
  - Update type hint if needed: `StrategyFactoryABC[..., CondensationStrategy]`
  - Test file: `particula/dynamics/condensation/tests/condensation_factories_test.py`
  - Test cases:
    - Factory returns staggered strategy via `"isothermal_staggered"`
    - Factory parameters propagate correctly to built strategy
    - Invalid strategy type raises `KeyError`

- [ ] **E1-F3-P3:** Export from `particula.dynamics` namespace with tests
  - Issue: TBD | Size: XS | Status: Not Started
  - Update `particula/dynamics/condensation/condensation_builder/__init__.py`:
    - Add `CondensationIsothermalStaggeredBuilder` to exports
  - Update `particula/dynamics/condensation/__init__.py`:
    - Add `CondensationIsothermalStaggered` to exports
    - Add `CondensationIsothermalStaggeredBuilder` to exports
  - Update `particula/dynamics/__init__.py`:
    - Add `CondensationIsothermalStaggered` to exports
    - Add `CondensationIsothermalStaggeredBuilder` to exports
  - Test file: `particula/dynamics/tests/imports_test.py` (or add to existing)
  - Smoke tests:
    - `from particula.dynamics import CondensationIsothermalStaggered`
    - `from particula.dynamics import CondensationIsothermalStaggeredBuilder`
    - `from particula.dynamics.condensation import CondensationIsothermalStaggered`
    - `import particula as par; par.dynamics.CondensationIsothermalStaggered`

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
"""Builder for the CondensationIsothermalStaggered strategy."""

from typing import Optional

from particula.abc_builder import BuilderABC
from particula.builder_mixin import BuilderMolarMassMixin
from particula.dynamics.condensation.condensation_strategies import (
    CondensationIsothermalStaggered,
    CondensationStrategy,
)

from .condensation_builder_mixin import (
    BuilderAccommodationCoefficientMixin,
    BuilderDiffusionCoefficientMixin,
    BuilderUpdateGasesMixin,
)


class CondensationIsothermalStaggeredBuilder(
    BuilderABC,
    BuilderMolarMassMixin,
    BuilderDiffusionCoefficientMixin,
    BuilderAccommodationCoefficientMixin,
    BuilderUpdateGasesMixin,
):
    """Fluent builder for :class:`CondensationIsothermalStaggered`.
    
    Extends the base condensation builder with staggered-stepping-specific
    parameters: theta_mode, num_batches, shuffle_each_step, and random_state.
    
    Example:
        >>> builder = CondensationIsothermalStaggeredBuilder()
        >>> strategy = (
        ...     builder
        ...     .set_molar_mass(0.018, "kg/mol")
        ...     .set_diffusion_coefficient(2e-5, "m^2/s")
        ...     .set_accommodation_coefficient(1.0)
        ...     .set_theta_mode("random")
        ...     .set_num_batches(10)
        ...     .build()
        ... )
    """

    def __init__(self) -> None:
        """Initialize the Condensation Isothermal Staggered builder."""
        required_parameters = [
            "molar_mass",
            "diffusion_coefficient",
            "accommodation_coefficient",
        ]
        BuilderABC.__init__(self, required_parameters)
        BuilderMolarMassMixin.__init__(self)
        BuilderDiffusionCoefficientMixin.__init__(self)
        BuilderAccommodationCoefficientMixin.__init__(self)
        BuilderUpdateGasesMixin.__init__(self)
        
        # Staggered-specific parameters with defaults
        self.theta_mode: str = "half"
        self.num_batches: int = 1
        self.shuffle_each_step: bool = True
        self.random_state: Optional[int] = None

    def set_theta_mode(
        self,
        theta_mode: str,
        theta_mode_units: Optional[str] = None,
    ) -> "CondensationIsothermalStaggeredBuilder":
        """Set the theta mode for staggered stepping.
        
        Args:
            theta_mode: One of "half", "random", or "batch".
            theta_mode_units: Ignored (for API consistency).
            
        Returns:
            Self for method chaining.
            
        Raises:
            ValueError: If theta_mode not in valid set.
        """
        valid_modes = {"half", "random", "batch"}
        if theta_mode not in valid_modes:
            raise ValueError(
                f"theta_mode must be one of {valid_modes}, got '{theta_mode}'"
            )
        self.theta_mode = theta_mode
        return self

    def set_num_batches(
        self,
        num_batches: int,
        num_batches_units: Optional[str] = None,
    ) -> "CondensationIsothermalStaggeredBuilder":
        """Set the number of Gauss-Seidel batches.
        
        Args:
            num_batches: Number of batches (>= 1).
            num_batches_units: Ignored (for API consistency).
            
        Returns:
            Self for method chaining.
            
        Raises:
            ValueError: If num_batches < 1.
        """
        if num_batches < 1:
            raise ValueError(f"num_batches must be >= 1, got {num_batches}")
        self.num_batches = num_batches
        return self

    def set_shuffle_each_step(
        self,
        shuffle: bool,
        shuffle_units: Optional[str] = None,
    ) -> "CondensationIsothermalStaggeredBuilder":
        """Set whether to shuffle particle order each step.
        
        Args:
            shuffle: If True, randomize particle order each step.
            shuffle_units: Ignored (for API consistency).
            
        Returns:
            Self for method chaining.
        """
        self.shuffle_each_step = shuffle
        return self

    def set_random_state(
        self,
        random_state: Optional[int],
        random_state_units: Optional[str] = None,
    ) -> "CondensationIsothermalStaggeredBuilder":
        """Set random state for reproducibility.
        
        Args:
            random_state: Seed for random number generator, or None.
            random_state_units: Ignored (for API consistency).
            
        Returns:
            Self for method chaining.
        """
        self.random_state = random_state
        return self

    def build(self) -> CondensationStrategy:
        """Validate parameters and create a staggered condensation strategy."""
        self.pre_build_check()

        # Type guards: pre_build_check ensures these are not None
        if self.diffusion_coefficient is None:
            raise ValueError("diffusion_coefficient must be set")
        if self.accommodation_coefficient is None:
            raise ValueError("accommodation_coefficient must be set")

        return CondensationIsothermalStaggered(
            molar_mass=self.molar_mass,
            diffusion_coefficient=self.diffusion_coefficient,
            accommodation_coefficient=self.accommodation_coefficient,
            update_gases=self.update_gases,
            theta_mode=self.theta_mode,
            num_batches=self.num_batches,
            shuffle_each_step=self.shuffle_each_step,
            random_state=self.random_state,
        )
```

### Factory Registration

```python
# In condensation_factories.py
"""Factory for building condensation strategies."""

from typing import Any, Dict

from particula.abc_factory import StrategyFactoryABC

from .condensation_builder.condensation_isothermal_builder import (
    CondensationIsothermalBuilder,
)
from .condensation_builder.condensation_isothermal_staggered_builder import (
    CondensationIsothermalStaggeredBuilder,
)
from .condensation_strategies import CondensationStrategy


class CondensationFactory(
    StrategyFactoryABC[CondensationIsothermalBuilder, CondensationStrategy]
):
    """Factory class for condensation strategies."""

    def get_builders(self) -> Dict[str, Any]:
        """Return the mapping of strategy types to builder instances.

        Returns:
            Dictionary mapping condensation strategy names to builders.
        """
        return {
            "isothermal": CondensationIsothermalBuilder(),
            "isothermal_staggered": CondensationIsothermalStaggeredBuilder(),
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

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Builder API inconsistency with existing builders | Low | Medium | Mirror exact pattern from `CondensationIsothermalBuilder` |
| Missing exports cause import errors | Low | High | Add comprehensive import smoke tests in E1-F3-P3 |
| Factory type hints become complex | Low | Low | Use `Any` for builder dict values if needed |

## Change Log

| Date       | Change                                | Author       |
|------------|---------------------------------------|--------------|
| 2025-12-23 | Initial feature documentation created | ADW Workflow |
| 2025-12-29 | Fixed file paths, added complete builder code, updated factory | ADW Workflow |

[epic-e1]: ../epics/E1-staggered-condensation-stepping.md
