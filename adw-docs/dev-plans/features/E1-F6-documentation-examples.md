# Feature E1-F6: Documentation and Examples

**Status:** Not Started
**Priority:** P3
**Assignees:** TBD
**Labels:** feature, dynamics, condensation, documentation, examples
**Milestone:** v0.3.x
**Size:** M (~150 LOC docs/examples)

**Start Date:** TBD
**Target Date:** TBD
**Created:** 2025-12-23
**Updated:** 2025-12-23

**Parent Epic:** [E1: Staggered ODE Stepping][epic-e1]
**Related Issues:** TBD
**Related PRs:** TBD

---

## Overview

Create comprehensive documentation and usage examples for the staggered
condensation stepping feature, including Google-style docstrings, a Jupyter
notebook tutorial, and development documentation updates.

### Problem Statement

Without documentation, users cannot:

- Understand when to use staggered vs simultaneous stepping
- Choose appropriate theta modes and batch sizes
- Verify their usage with working examples
- Find the feature in API documentation

### Value Proposition

- **Discoverability**: Users find the feature through docs and examples
- **Adoption**: Working examples reduce barrier to entry
- **Correctness**: Documentation prevents misuse
- **Maintenance**: Future developers understand design decisions

## Scope

### In Scope

- Google-style docstrings for all new public methods
- Literature citations in module docstring
- Jupyter notebook example demonstrating all modes
- Mass conservation verification in example
- Development documentation updates (epic status, lessons learned)

### Out of Scope

- Core staggered stepping logic (E1-F1, E1-F2)
- Builder and factory integration (E1-F3)
- Mass conservation validation (E1-F4)
- Performance benchmarks (E1-F5)
- API reference page generation (automated from docstrings)

## Dependencies

### Upstream

- **E1-F1**: Core Staggered Stepping Logic (must be complete)
- **E1-F2**: Batch-Wise Stepping Mode (must be complete)
- **E1-F3**: Builder and Factory Integration (must be complete)
- **E1-F4**: Mass Conservation Validation (recommended)
- **E1-F5**: Stability and Performance Benchmarks (recommended)

### Downstream

- None (final feature in epic)

## Phase Checklist

- [ ] **E1-F6-P1:** Add docstrings and inline documentation with validation
  - Issue: TBD | Size: S | Status: Not Started
  - Google-style docstrings for all new public methods
  - Add literature citations in module docstring
  - Run docstring linter to validate formatting
  - Include docstring coverage in test suite

- [ ] **E1-F6-P2:** Create usage example notebook
  - Issue: TBD | Size: M | Status: Not Started
  - File: `docs/Examples/Dynamics/staggered_condensation_example.ipynb`
  - Demonstrate all three modes (half, random, batch)
  - Show mass conservation verification
  - Compare stability with simultaneous stepping
  - Include visualization of results

- [ ] **E1-F6-P3:** Update development documentation
  - Issue: TBD | Size: XS | Status: Not Started
  - Update `adw-docs/dev-plans/README.md` with epic status
  - Update `adw-docs/dev-plans/epics/index.md` with completion
  - Add completion notes and lessons learned to epic document

## Critical Testing Requirements

- **Docstring Validation**: Run docstring linter on all new code
- **Notebook Execution**: Notebook must execute without errors
- **Link Validation**: All internal links must resolve
- **Example Correctness**: Example outputs must match expected behavior

## Documentation Strategy

### Docstring Standards

Location: `particula/dynamics/condensation/condensation_strategies.py`

**Required Docstrings:**

- [ ] Module docstring with literature citations
- [ ] `CondensationIsothermalStaggered` class docstring
- [ ] `__init__` method docstring with all parameters
- [ ] `step()` method docstring with algorithm description
- [ ] `_get_theta_values()` method docstring
- [ ] `_make_batches()` method docstring

### Example Notebook Structure

Location: `docs/Examples/Dynamics/staggered_condensation_example.ipynb`

**Sections:**

1. Introduction and motivation
2. Setup: Create particle system and gas species
3. Mode comparison: half vs random vs batch
4. Mass conservation verification
5. Stability comparison with simultaneous stepping
6. Choosing parameters: theta mode and batch count
7. Performance considerations

### Development Documentation Updates

**Files to Update:**

- `adw-docs/dev-plans/README.md` - Update epic status
- `adw-docs/dev-plans/epics/index.md` - Mark E1 complete
- `adw-docs/dev-plans/epics/E1-staggered-condensation-stepping.md` - Add
  completion notes

## Technical Approach

### Module Docstring Template

```python
"""Staggered isothermal condensation strategies.

This module provides staggered ODE stepping for particle-resolved condensation
simulations. The staggered approach improves numerical stability and mass
conservation compared to simultaneous particle updates.

The implementation draws on established numerical methods:

1. **Operator Splitting**: LeVeque (2002) — Finite Volume Methods for
   Hyperbolic Problems
2. **Symplectic Integration**: Hairer, Lubich, & Wanner (2006) — Geometric
   Numerical Integration
3. **Mass-Conserving Condensation**: Jacobson (1997, 1999) — Analytical
   Predictor of Condensation (APC)
4. **Particle-Resolved Modeling**: Riemer et al. (2009) — PartMC-MOSAIC

Example:
    >>> from particula.dynamics.condensation import (
    ...     CondensationIsothermalStaggered,
    ... )
    >>> strategy = CondensationIsothermalStaggered(
    ...     molar_mass=0.018,
    ...     theta_mode="random",
    ...     num_batches=10,
    ... )
    >>> particle, gas = strategy.step(
    ...     particle, gas_species, 298.0, 101325.0, 0.1
    ... )
"""
```

### Class Docstring Template

```python
class CondensationIsothermalStaggered(CondensationStrategy):
    """Staggered isothermal condensation with configurable stepping modes.

    This strategy implements a two-pass staggered stepping algorithm that
    preserves mass conservation and improves numerical stability for
    particle-resolved condensation simulations.

    Three stepping modes are available:

    - ``"half"``: Deterministic half-step (theta = 0.5 for all particles)
    - ``"random"``: Randomized fractional step (theta ~ U[0,1] per particle)
    - ``"batch"``: Gauss-Seidel batch updates with configurable batch count

    Args:
        molar_mass: Molar mass of condensing species [kg/mol].
        theta_mode: Stepping mode, one of "half", "random", "batch".
            Defaults to "half".
        num_batches: Number of batches for batch-wise stepping.
            Defaults to 1 (no batching).
        shuffle_each_step: Whether to shuffle particle order each step.
            Only affects "random" mode. Defaults to True.

    Raises:
        ValueError: If theta_mode is not one of the valid options.
        ValueError: If num_batches < 1.

    Example:
        >>> strategy = CondensationIsothermalStaggered(
        ...     molar_mass=0.018,
        ...     theta_mode="random",
        ... )
        >>> particle, gas = strategy.step(
        ...     particle, gas_species, 298.0, 101325.0, time_step=0.1
        ... )

    Note:
        For best mass conservation, use "random" mode with num_batches >= 10.
        The "half" mode is fastest but may show slight systematic bias.

    See Also:
        CondensationIsothermal: Simultaneous (non-staggered) stepping.
        CondensationIsothermalStaggeredBuilder: Fluent builder for this class.
    """
```

### Notebook Example Outline

```python
# Cell 1: Introduction
"""
# Staggered Condensation Stepping

This notebook demonstrates the staggered ODE stepping framework for
particle-resolved condensation simulations. We'll compare three stepping
modes and verify mass conservation.
"""

# Cell 2: Setup
import numpy as np
import matplotlib.pyplot as plt
from particula.dynamics.condensation import CondensationIsothermalStaggered
# ... particle and gas setup

# Cell 3: Mode Comparison
modes = ["half", "random", "batch"]
results = {}
for mode in modes:
    strategy = CondensationIsothermalStaggered(
        molar_mass=0.018,
        theta_mode=mode,
        num_batches=10,
    )
    results[mode] = run_simulation(strategy)

# Cell 4: Mass Conservation
def verify_mass_conservation(initial, final):
    # ... calculate and plot mass over time

# Cell 5: Stability Comparison
# ... compare with simultaneous stepping at large dt

# Cell 6: Parameter Guidance
# ... recommendations for theta_mode and num_batches
```

## Success Criteria

- [ ] All new public methods have Google-style docstrings
- [ ] Module docstring includes literature citations
- [ ] Docstring linter passes without errors
- [ ] Jupyter notebook executes without errors
- [ ] Notebook demonstrates all three theta modes
- [ ] Mass conservation verification included in notebook
- [ ] Development documentation updated with epic status
- [ ] All internal links resolve correctly

## Usage Example

```python
# Access documentation
from particula.dynamics.condensation import CondensationIsothermalStaggered
help(CondensationIsothermalStaggered)

# Run example notebook
jupyter notebook docs/Examples/Dynamics/staggered_condensation_example.ipynb
```

## Change Log

| Date       | Change                                | Author       |
|------------|---------------------------------------|--------------|
| 2025-12-23 | Initial feature documentation created | ADW Workflow |

[epic-e1]: ../epics/E1-staggered-condensation-stepping.md
