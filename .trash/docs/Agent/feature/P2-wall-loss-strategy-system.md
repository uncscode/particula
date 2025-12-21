# Feature: Wall Loss Strategy System (WallLossStrategy & SphericalWallLossStrategy)

**Status:** In Progress
**Priority:** P2
**Assignees:** ADW Workflow
**Labels:** feature, dynamics, wall-loss
**Milestone:** v0.2.x
**Size:** M (~100 LOC core, plus tests)

**Start Date:** 2025-12-02
**Target Date:** 2025-12-31
**Created:** 2025-12-02
**Updated:** 2025-12-02

**Related Issues:** #816
**Related PRs:** (pending)
**Related ADRs:** (none yet)

---

## Overview

This feature introduces a **wall loss strategy system** that aligns chamber wall
loss with the existing **dynamics strategy/builder pattern** used for
condensation and coagulation. Instead of calling standalone wall loss rate
functions directly, users can now work with a unified, object-oriented API
based on a `WallLossStrategy` abstract base class and a
`SphericalWallLossStrategy` concrete implementation.

The new strategy classes operate on `ParticleRepresentation` objects and
support all three distribution types used throughout particula dynamics:

- `"discrete"`
- `"continuous_pdf"`
- `"particle_resolved"`

The implementation reuses existing wall loss coefficient and rate utilities and
exports the new strategies through `particula.dynamics`, keeping the public API
coherent with other dynamics modules.

See also the wall loss overview in the main documentation index
([Dynamics and wall loss](../../index.md)) and the
[Chamber wall loss end-to-end example](../../Examples/Chamber_Wall_Loss/index.md).

### Problem Statement

Previously, wall loss was only available through standalone functions in
`particula/dynamics/wall_loss.py`. This made it harder to integrate wall loss
with the rest of the dynamics system, which is built around strategy classes
and builders that operate on `ParticleRepresentation`.

Without a strategy-based wall loss API:

- Users had to learn a separate, function-only interface.
- Integrating wall loss with other processes (coagulation, condensation,
  dilution) required custom glue code.
- There was no clear abstraction for adding new wall loss models or geometries.

### Value Proposition

The wall loss strategy system provides:

- **Consistent API** with other dynamics strategies (`rate`, `step`,
  `distribution_type`).
- **Better integration** with `ParticleRepresentation` and existing builders.
- **Extensibility**: new geometries or models can be added as additional
  `WallLossStrategy` implementations.
- **Re-use of physics**: existing wall loss coefficient utilities remain the
  single source of truth for underlying calculations.

## Phases

> **Philosophy:** Each phase should be one GitHub issue with a focused scope
> (~100 lines of code or less, excluding tests/docs). Small, focused changes
> make reviews smooth and safe. Smooth is safe, and safe is fast.

- [x] **Phase 1:** Introduce wall loss strategy system
  - GitHub Issue: #816
  - Status: In Progress (implementation and tests in review)
  - Size: M (~100 LOC core strategies + tests)
  - Dependency: Existing wall loss coefficient utilities
  - Estimated Effort: 1–2 days

## User Stories

### Story 1: Strategy-based wall loss for dynamics users
**As a** particula dynamics user
**I want** to apply wall loss using the same strategy API as coagulation and
condensation
**So that** I can compose wall loss with other processes in a consistent way.

**Acceptance Criteria:**
- [ ] `WallLossStrategy` defines a documented interface consistent with other
      dynamics strategies.
- [ ] `SphericalWallLossStrategy` is available via `particula.dynamics`.
- [ ] Example usage composes wall loss with other dynamics without custom
      glue code.

### Story 2: Spherical chamber wall loss with all distribution types
**As a** chamber simulation developer
**I want** a spherical wall loss strategy that works for `"discrete"`,
`"continuous_pdf"`, and `"particle_resolved"` distributions
**So that** I can reuse the same API across different modeling regimes.

**Acceptance Criteria:**
- [ ] `SphericalWallLossStrategy.step` supports all three distribution types
      without errors.
- [ ] Loss rates are negative and reduce concentration for discrete and
      continuous PDF distributions.
- [ ] Particle-resolved simulations see a physically reasonable reduction in
      effective concentration.

## Technical Approach

### Architecture Changes

The implementation plan (Issue #816) introduces a wall loss strategy system in
six main steps:

1. **Refactor wall loss into a package**
   - Convert `particula/dynamics/wall_loss.py` into a package
     `particula/dynamics/wall_loss/`.
   - Move legacy rate functions into `rate.py` and re-export them from the
     package `__init__`.

2. **Introduce `WallLossStrategy` ABC**
   - New abstract base class in
     `particula/dynamics/wall_loss/wall_loss_strategies.py`.
   - Encapsulates distribution handling, rate calculation, and time stepping
     on `ParticleRepresentation`.

3. **Implement `SphericalWallLossStrategy`**
   - Concrete strategy that uses existing spherical wall loss coefficient
     utilities and chamber geometry parameters.

4. **Export via `particula.dynamics`**
   - `WallLossStrategy` and `SphericalWallLossStrategy` are re-exported so that
     users access them as `particula.dynamics.WallLossStrategy` and
     `particula.dynamics.SphericalWallLossStrategy`.

5. **Add unit tests**
   - Focused tests live under
     `particula/dynamics/wall_loss/tests/wall_loss_strategies_test.py`.

6. **Sanity-check integration**
   - Ensure legacy wall loss tests continue to pass and that strategy-based
     and function-based APIs are numerically consistent.

**Affected Components:**
- `particula.dynamics.wall_loss` – refactored into a package containing both
  legacy rate functions and new strategy classes.
- `particula.dynamics` – extended namespace to expose
  `WallLossStrategy` and `SphericalWallLossStrategy` alongside other
  dynamics strategies.
- `particula.dynamics.properties.wall_loss_coefficient` – reused by
  strategies for underlying physics (no behavior change).

### Design Patterns

- **Strategy pattern** for wall loss models: `WallLossStrategy` defines the
  interface, `SphericalWallLossStrategy` is the first concrete implementation.
- **Abstract base class (ABC)** via `abc.ABC` and `@abstractmethod` to enforce
  implementation of the core `loss_coefficient` method.
- **Validated initialization** using `@validate_inputs` for physical
  parameters (e.g., wall eddy diffusivity, chamber radius).

### WallLossStrategy API

`WallLossStrategy` is the abstract base class for all wall loss strategies.
It is not instantiated directly.

**Key attributes:**

- `wall_eddy_diffusivity: float` – wall eddy diffusivity used in coefficient
  calculations.
- `distribution_type: str` – one of:
  - `"discrete"`
  - `"continuous_pdf"`
  - `"particle_resolved"`

**Key methods:**

- `loss_coefficient(self, particle, temperature, pressure)` (abstract)
  - Returns the wall loss coefficient  for the current strategy.
- `loss_rate(self, particle, temperature, pressure)`
  - Computes `k = loss_coefficient(...)` and obtains concentration via
    `particle.get_concentration()`.
  - Returns `-k * concentration` (negative sign encodes loss).
- `rate(self, particle, temperature, pressure)`
  - Thin wrapper returning `np.asarray(self.loss_rate(...))` for consistent
    array types.
- `step(self, particle, temperature, pressure, time_step)`
  - Applies a first-order wall loss update over `time_step`.

**Distribution handling in `step`:**

- For `"discrete"` and `"continuous_pdf"` distributions:
  - Compute `rate = self.rate(...)`.
  - Call `particle.add_concentration(rate * time_step)`.
  - Return the updated `ParticleRepresentation`.

- For `"particle_resolved"` distributions:
  - Interpret wall loss as a first-order removal process at the particle
    level.
  - Use the rate to derive a survival probability over the time step and
    reduce effective concentration accordingly (e.g., via a Bernoulli-like
    mask or deterministic approximation that preserves the distribution
    shape).

Invalid `distribution_type` values raise `ValueError` with a clear error
message, mirroring existing coagulation strategy behavior.

### SphericalWallLossStrategy

`SphericalWallLossStrategy` is the first concrete implementation of
`WallLossStrategy`. It models wall loss in a **spherical chamber** using
existing coefficient utilities.

**Initialization:**

- `wall_eddy_diffusivity: float` – positive wall eddy diffusivity .
- `chamber_radius: float` – positive chamber radius .
- `distribution_type: str` – distribution type ("discrete",
  "continuous_pdf", or "particle_resolved"), defaulting to "discrete".

The `loss_coefficient` implementation delegates to
`get_spherical_wall_loss_coefficient_via_system_state` with arguments:

- `wall_eddy_diffusivity=self.wall_eddy_diffusivity`
- `particle_radius=particle.get_radius()`
- `particle_density=particle.get_effective_density()`
- `temperature=temperature`
- `pressure=pressure`
- `chamber_radius=self.chamber_radius`

The resulting coefficient is then used by the base class methods to compute
loss rates and apply time stepping.

**Example usage (mirrors AGENTS.md and issue description):**

```python
import particula as par

strategy = par.dynamics.SphericalWallLossStrategy(
    wall_eddy_diffusivity=0.001,
    chamber_radius=0.5,
    distribution_type="discrete",
)

rate = strategy.rate(
    particle=particle,
    temperature=298.0,
    pressure=101325.0,
)

particle = strategy.step(
    particle=particle,
    temperature=298.0,
    pressure=101325.0,
    time_step=1.0,
)
```

The full API for `WallLossStrategy` and `SphericalWallLossStrategy` is
documented in their class docstrings. MkDocStrings will generate API reference
pages from these docstrings in the online documentation.

## Implementation Tasks

### Core dynamics and API
- [x] Refactor `particula.dynamics.wall_loss` into a package with `rate.py`.
- [x] Add `WallLossStrategy` ABC operating on `ParticleRepresentation`.
- [x] Implement `SphericalWallLossStrategy` using existing coefficient
      utilities.
- [x] Wire `WallLossStrategy` and `SphericalWallLossStrategy` into the
      `particula.dynamics` namespace.

### Testing
- [x] Add unit tests under
      `particula/dynamics/wall_loss/tests/wall_loss_strategies_test.py`.
- [x] Test invalid `distribution_type` handling.
- [x] Test that `rate` is negative and `step` reduces concentration for
      discrete and continuous PDF distributions.
- [x] Test that particle-resolved distributions are handled without errors and
      reduce effective concentration.
- [x] Keep existing wall loss coefficient and rate tests passing.

### Documentation
- [x] Add this feature document in `docs/Agent/feature/`.
- [ ] Ensure main docs index and README wall loss sections match the new API.
- [ ] Add or update an example notebook in
      `docs/Examples/Chamber_Wall_Loss/` if needed.

## Dependencies

### Upstream Dependencies
- `ParticleRepresentation` – strategy operates on particle distributions.
- `particula.dynamics.properties.wall_loss_coefficient` – provides spherical
  wall loss coefficient calculations.

### Downstream Dependencies
- Chamber wall loss examples and notebooks under
  `docs/Examples/Chamber_Wall_Loss/`.
- Any future wall loss strategies that subclass `WallLossStrategy`.

### External Dependencies
- None beyond existing particula runtime dependencies.

## Blockers

- [ ] None identified at this time.

## Testing Strategy

### Unit Tests

Unit tests focus on:

- ABC behavior (non-instantiability of `WallLossStrategy`).
- Validation of `distribution_type` values.
- Parameter storage and coefficient computation in
  `SphericalWallLossStrategy`.
- Sign and magnitude checks for loss rates.

**Test Cases:**
- [x] `WallLossStrategy` cannot be instantiated directly.
- [x] Invalid `distribution_type` raises `ValueError`.
- [x] `loss_coefficient` is positive for reasonable physical parameters.
- [x] `rate` is negative and `step` reduces concentration for discrete and
      continuous PDF distributions.
- [x] Zero-concentration edge cases leave concentration unchanged.

### Integration Tests

Integration tests (optional but recommended):

- Compare mean rates from `SphericalWallLossStrategy.rate` against
  `get_spherical_wall_loss_rate` for matching configurations.
- Compose wall loss with other dynamics strategies (e.g., condensation) in a
  small chamber simulation.

**Test Cases:**
- [ ] Strategy-based rate matches function-based rate within tolerance.
- [ ] Combined dynamics + wall loss simulation runs without errors.

### Performance Tests

- [ ] (Optional) Benchmark wall loss strategies on large particle-resolved
      systems to ensure performance is acceptable and comparable to the
      function-based API.

## Documentation

- [x] Feature documentation (this file).
- [ ] Cross-check wall loss sections in `docs/index.md` and `readme.md`.
- [ ] Verify that mkdocstrings generates API reference pages for
      `WallLossStrategy` and `SphericalWallLossStrategy`.

## Security Considerations

Wall loss strategies operate entirely on in-memory simulation data and do not
introduce I/O, network access, or external integration. No additional security
concerns are expected beyond standard Python safety practices.

## Performance Considerations

- Strategy-based wall loss should be comparable in performance to the existing
  function-based API.
- Additional overhead from strategy objects should be minimal relative to the
  cost of coefficient and rate calculations.
- Particle-resolved handling must avoid unnecessary allocations inside tight
  loops.

## Rollout Strategy

### Deployment Plan

- Land refactor and new strategies behind the existing
  `particula.dynamics.wall_loss` public API.
- Keep legacy rate functions available and behavior-compatible.
- Encourage new code and examples to use `SphericalWallLossStrategy`.

### Rollback Plan

- If issues arise, revert the strategy module while leaving the refactored
  package structure intact, or fully revert to the previous flat module layout
  if necessary.

## Success Criteria

- [ ] `particula.dynamics.wall_loss` is a package providing both legacy rate
      functions and new strategy classes.
- [ ] `WallLossStrategy` ABC defines a clear, documented interface and cannot
      be instantiated directly.
- [ ] `SphericalWallLossStrategy` correctly computes coefficients using
      `get_spherical_wall_loss_coefficient_via_system_state`.
- [ ] `SphericalWallLossStrategy.step` handles all supported distribution types
      without errors and reduces concentration where appropriate.
- [ ] New unit tests achieve high coverage for wall loss strategies.
- [ ] Existing wall loss tests continue to pass.
- [ ] Users can access strategies via `particula.dynamics` and examples
      demonstrate their use.

## Metrics to Track

- Wall loss strategy test coverage (lines and branches).
- Runtime performance relative to function-based wall loss API.
- Usage of strategy-based API in examples and downstream projects.

## Timeline

| Phase/Milestone | Start Date  | Target Date | Actual Date | Status        |
|-----------------|-------------|-------------|-------------|---------------|
| Phase 1         | 2025-12-02  | 2025-12-31  | —           | In Progress   |

## Open Questions

- [ ] Which additional chamber geometries (e.g., rectangular) should be
      promoted to first-class strategies?
- [ ] Should there be a higher-level builder for composing multiple wall loss
      strategies?

## References

- Existing wall loss overview and examples in `docs/index.md` and
  `docs/Examples/Chamber_Wall_Loss/`.
- API documentation generated by mkdocstrings for
  `particula.dynamics.WallLossStrategy` and
  `particula.dynamics.SphericalWallLossStrategy`.

## Change Log

| Date       | Change                              | Author        |
|------------|-------------------------------------|---------------|
| 2025-12-02 | Initial feature documentation added | ADW Workflow  |
