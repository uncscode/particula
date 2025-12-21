# ADR-001: Strategy-based wall loss subsystem and `wall_loss` package refactor

**Status:** Accepted
**Date:** 2025-12-02
**Decision Makers:** ADW Development Team
**Technical Story:** #816

## Context

Existing wall loss functionality in `particula.dynamics` was implemented primarily as a set of standalone rate functions in `particula/dynamics/wall_loss.py`. While this API worked for simple use cases, it diverged from the strategy-based design already used for condensation and coagulation. As a result, wall loss could not be configured or composed in the same way as other dynamic processes, and reusing common abstractions (builders, strategies, particle representations) was harder than necessary.

At the same time, we want to preserve the existing function-based API so that users and tests depending on `get_spherical_wall_loss_rate` and `get_rectangle_wall_loss_rate` continue to work without modification.

### Problem Statement

How can we introduce a flexible, strategy-based wall loss subsystem that aligns with existing dynamics patterns (condensation and coagulation strategies), while preserving the current function-based API and clarifying the module structure around wall loss?

### Forces

**Driving Forces:**
- Align wall loss with existing strategy-based patterns in `dynamics` for consistency
- Enable reusable, object-oriented wall loss models that operate directly on `ParticleRepresentation`
- Clarify module structure by separating legacy rate functions from higher-level strategies

**Restraining Forces:**
- Need to preserve backwards compatibility for existing imports and tests
- Avoid introducing circular imports between `dynamics`, `particles`, and `util`
- Keep the public API simple for users who only need scalar rate functions

## Decision

We will introduce a strategy-based wall loss subsystem focused on a new `WallLossStrategy` abstract base class and a concrete `SphericalWallLossStrategy`, and we will refactor `particula.dynamics.wall_loss` from a single module into a package that hosts both the legacy rate functions and the new strategy classes.

Concretely:

1. Refactor `particula/dynamics/wall_loss.py` into a `particula/dynamics/wall_loss/` package.
2. Move the existing rate functions into `particula/dynamics/wall_loss/rate.py` and re-export them via the `wall_loss` package.
3. Introduce `WallLossStrategy` and `SphericalWallLossStrategy` in `particula/dynamics/wall_loss/wall_loss_strategies.py`, operating directly on `ParticleRepresentation` and supporting all distribution types.
4. Export the new strategies via both `particula.dynamics.wall_loss` and the `particula.dynamics` namespace to match other dynamics strategies.
5. Add targeted tests under `particula/dynamics/wall_loss/tests/` to validate the new strategy behavior and maintain coverage of legacy functions.

### Chosen Option

**Option 2: Strategy-based wall loss package (functions + strategies)**

We adopt a hybrid approach where the `wall_loss` functionality is organized as a package. The package contains a `rate.py` module with the existing scalar rate functions and a `wall_loss_strategies.py` module with the new strategy classes. The package `__init__.py` re-exports both the legacy functions and the new strategies, and `particula.dynamics.__init__` provides convenient namespace exports for the strategies.

This keeps the function-based interface stable while enabling an extensible, strategy-oriented design for new wall loss models.

## Alternatives Considered

### Option 1: Keep `wall_loss.py` as a single module with only functions

**Description:**
Continue to expose only function-style wall loss APIs in a single `wall_loss.py` module and add any new behavior as additional functions.

**Pros:**
- No structural refactor required
- Minimal implementation effort
- Zero risk of breaking imports due to package/module changes

**Cons:**
- Inconsistent with strategy-based design used for condensation and coagulation
- Harder to encapsulate stateful configuration (e.g., chamber geometry) in reusable objects
- Difficult to operate directly on `ParticleRepresentation` and distribution types in a unified way

**Reason for Rejection:**
This option fails to align wall loss with the existing strategy-based architecture in `dynamics`, limiting extensibility and composability for future wall loss models.

---

### Option 2: Strategy-only refactor (replace functions with strategies)

**Description:**
Replace the existing function-based wall loss API with strategy objects only (e.g., `SphericalWallLossStrategy`), deprecating or removing direct rate functions.

**Pros:**
- Strong architectural consistency with other strategy-based subsystems
- Clear, object-oriented interface around wall loss models
- Encourages use of `ParticleRepresentation` and distribution-aware APIs

**Cons:**
- Breaks existing users and tests that import and call legacy rate functions
- Requires a more complex migration path and deprecation process
- Higher initial migration cost for downstream consumers

**Reason for Rejection:**
While architecturally clean, this option introduces unnecessary breaking changes. We prefer an incremental migration path that preserves existing function-based APIs.

---

### Option 3: New wall loss process type without refactoring `wall_loss`

**Description:**
Introduce a new `WallLoss` process class (similar to `MassCondensation` and `Coagulation`) that internally uses the existing rate functions, without refactoring `wall_loss` into a package or introducing a dedicated strategy abstraction.

**Pros:**
- Provides a process-level abstraction for wall loss
- Limited changes to existing modules
- Keeps the function-based implementation intact

**Cons:**
- Adds another layer without solving the lack of a shared strategy abstraction
- Harder to reuse configuration and behavior across different wall loss models
- Does not align with the existing strategy/builder/factory patterns in `dynamics`

**Reason for Rejection:**
This option addresses only the process-level API and not the underlying need for a reusable, strategy-based wall loss abstraction that works consistently with `ParticleRepresentation`.

---

## Rationale

### Why This Approach?

The chosen hybrid approach (package containing both functions and strategies) balances architectural consistency with backwards compatibility:

1. **Consistency with existing patterns**
   - Aligns wall loss with the strategy-based design used for condensation and coagulation.
   - Enables future wall loss strategies to follow the same ABC/builder/factory conventions.

2. **Backwards compatibility**
   - Keeps `get_spherical_wall_loss_rate` and `get_rectangle_wall_loss_rate` available at their existing import paths via the new package layout.
   - Minimizes disruption to existing user code and tests.

3. **Clear module boundaries**
   - Separates low-level rate calculations (`rate.py`) from high-level, `ParticleRepresentation`-aware strategies (`wall_loss_strategies.py`).
   - Makes it easier to discover wall loss-related functionality under `particula.dynamics.wall_loss`.

### Trade-offs Accepted

1. **Slightly more complex package structure**: Introducing a package and multiple modules increases structural complexity but provides clearer separation of concerns.
2. **Two parallel APIs (functions and strategies)**: Maintaining both scalar functions and strategy objects adds duplication risk, but we accept this to preserve compatibility and provide a smooth migration path.
3. **Additional dependency coupling**: Strategies depend on `ParticleRepresentation` and wall loss coefficient utilities, which slightly tightens the coupling between `particles` and `dynamics`, but this is consistent with other dynamics strategies.

## Consequences

### Positive

- Wall loss behavior can be expressed via reusable strategies that work directly with `ParticleRepresentation` and its distribution types.
- The `dynamics` subsystem has a more uniform API surface: condensation, coagulation, and wall loss all expose strategy-based abstractions.
- The `wall_loss` package clearly organizes related functionality (legacy rate functions, strategies, tests).

### Negative

- Developers must understand both the legacy function API and the new strategy-based API.
- The package/module refactor could cause issues for code relying on implicit module layout assumptions (e.g., deep imports), though public imports are preserved.

### Neutral

- Users who only rely on scalar rate functions see no behavioral change but benefit from clearer documentation and structure.

## Implementation

### Required Changes

1. **Refactor `wall_loss` into a package**
   - Create `particula/dynamics/wall_loss/` and move existing contents of `wall_loss.py` into `wall_loss/rate.py`.
   - Add `particula/dynamics/wall_loss/__init__.py` to re-export:
     - `get_spherical_wall_loss_rate`
     - `get_rectangle_wall_loss_rate`
   - Affected files: `particula/dynamics/wall_loss.py` (removed), `particula/dynamics/wall_loss/__init__.py`, `particula/dynamics/wall_loss/rate.py`.
   - Estimated effort: Low–medium.

2. **Introduce strategy abstractions for wall loss**
   - Implement `WallLossStrategy` ABC and `SphericalWallLossStrategy` in `particula/dynamics/wall_loss/wall_loss_strategies.py`.
   - Ensure strategies operate on `ParticleRepresentation` and support `"discrete"`, `"continuous_pdf"`, and `"particle_resolved"` distribution types.
   - Export strategies from `particula.dynamics.wall_loss` and `particula.dynamics`.
   - Affected files: `particula/dynamics/wall_loss/wall_loss_strategies.py`, `particula/dynamics/wall_loss/__init__.py`, `particula/dynamics/__init__.py`.
   - Estimated effort: Medium.

3. **Add tests and documentation updates**
   - Add `particula/dynamics/wall_loss/tests/wall_loss_strategies_test.py` and supporting fixtures.
   - Update architecture documentation to describe the wall loss package and strategy-based API.
   - Affected files: `particula/dynamics/wall_loss/tests/wall_loss_strategies_test.py`, `docs/Agent/architecture/architecture_outline.md`, `docs/Agent/architecture/architecture_guide.md`, `docs/Agent/architecture/decisions/README.md`.
   - Estimated effort: Medium.

### Migration Plan

1. Introduce the `wall_loss` package and move the legacy rate functions into `rate.py`, re-exporting them via `__init__.py` so that existing imports continue to work.
2. Implement `WallLossStrategy` and `SphericalWallLossStrategy` and export them via `particula.dynamics.wall_loss` and `particula.dynamics`.
3. Add unit tests for the new strategies and ensure existing wall loss–related tests continue to pass.
4. Update architecture documentation (outline, guide, and ADR index) to describe the new package structure and strategy-based API.
5. Encourage new code to use `SphericalWallLossStrategy` via documentation and examples, keeping legacy functions for backwards compatibility.

### Testing Strategy

- Extend the existing dynamics test suite with `wall_loss_strategies_test.py` to cover:
  - ABC behavior (`WallLossStrategy` cannot be instantiated directly).
  - Distribution type validation and error handling.
  - Correct computation of wall loss coefficients and rates for `SphericalWallLossStrategy`.
  - Consistent behavior across `"discrete"`, `"continuous_pdf"`, and `"particle_resolved"` distributions.
- Continue to run existing wall loss coefficient and rate tests to ensure backwards compatibility.
- Add optional integration tests comparing `SphericalWallLossStrategy.rate` with `get_spherical_wall_loss_rate` under matched conditions.

### Rollback Plan

If issues arise with the new package or strategies:

1. Temporarily stop exporting `WallLossStrategy` and `SphericalWallLossStrategy` from `particula.dynamics` and `particula.dynamics.wall_loss`.
2. Restore the previous `particula/dynamics/wall_loss.py` module from version control and deprecate the package structure.
3. Keep tests for legacy rate functions to ensure their behavior remains stable.
4. Revisit the design to address any discovered issues (e.g., coupling, performance, or API ergonomics) before attempting a revised strategy-based implementation.

## Validation

### Success Criteria

- [ ] `particula.dynamics.wall_loss` is a package that re-exports existing rate functions without breaking their public import paths.
- [ ] `WallLossStrategy` and `SphericalWallLossStrategy` are available via `particula.dynamics` and operate correctly on `ParticleRepresentation` for all supported distribution types.
- [ ] New wall loss strategy tests pass and achieve high coverage of `wall_loss_strategies.py`.
- [ ] Existing wall loss–related tests continue to pass without modification.
- [ ] Architecture documentation describes the wall loss package and strategy-based API.

### Metrics

- **Test coverage:** >90% line coverage for `wall_loss_strategies.py`.
- **Backward compatibility:** Zero breaking changes for documented public wall loss function imports.
- **Adoption:** New examples and features use `SphericalWallLossStrategy` rather than adding new standalone rate functions.

## References

### Related ADRs

- None yet.

### External References

- Crump, J. G., & Seinfeld, J. H. (1981). Turbulent deposition and gravitational sedimentation of an aerosol in a vessel of arbitrary shape. *Journal of Colloid and Interface Science*.  
- Existing particula documentation on wall loss coefficients and dynamics patterns.

### Documentation Updates

Files updated as part of this decision:
- [x] `docs/Agent/architecture/architecture_outline.md`
- [x] `docs/Agent/architecture/architecture_guide.md`
- [x] `docs/Agent/architecture/decisions/README.md`
- [x] `docs/Agent/architecture/decisions/ADR-001-strategy-based-wall-loss-subsystem.md`

## Notes

This ADR establishes the pattern for strategy-based wall loss models and the `wall_loss` package structure. Future wall loss geometries (e.g., rectangular chambers) should follow the same pattern by implementing additional `WallLossStrategy` subclasses and exporting them via the `wall_loss` package and `particula.dynamics`.
