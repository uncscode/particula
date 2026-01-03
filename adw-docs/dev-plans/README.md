# Development Plans

Feature and maintenance development plans tracked for particula. Each entry
follows the appropriate template to capture motivation, scope, phases, testing,
and rollout.

## Indexes

- [Epics Index](epics/index.md) — All epics with status and next available ID
- [Features Index](features/index.md) — All features with status and next
  available ID
- [Maintenance Index](maintenance/index.md) — All maintenance plans with status

## Epics

- [E1: Staggered ODE Stepping for Particle-Resolved Condensation][epic-e1]
  — Status: Completed (P2)
  - Scope: Staggered ODE stepping framework for particle-resolved condensation
    with three modes (half-step, random, batch) for improved stability and
    mass conservation.
  - Features: [E1-F1][e1-f1], [E1-F2][e1-f2], [E1-F3][e1-f3], [E1-F4][e1-f4],
    [E1-F5][e1-f5], [E1-F6][e1-f6]

## Feature Plans

### Epic E1: Staggered Condensation Features

- [E1-F1: Core Staggered Stepping Logic][e1-f1] — Status: Completed (P2)
  - Scope: `CondensationIsothermalStaggered` class with theta modes and two-pass
    stepping algorithm.
- [E1-F2: Batch-Wise Stepping Mode][e1-f2] — Status: Completed (P2)
  - Scope: Gauss-Seidel batch stepping with configurable batch count.
- [E1-F3: Builder and Factory Integration][e1-f3] — Status: Completed (P2)
  - Scope: Builder and factory support for staggered condensation strategy.
- [E1-F4: Mass Conservation Validation][e1-f4] — Status: Completed (P2)
  - Scope: Comprehensive test suites validating mass conservation properties.
- [E1-F5: Stability and Performance Benchmarks][e1-f5] — Status: Completed
  (P3)
  - Scope: Benchmark tests for stability and performance characteristics (P2 performance benchmarks landed; issue #137).
- [E1-F6: Documentation and Examples][e1-f6] — Status: Completed (P3)
  - Scope: Docstrings, Jupyter notebook examples, and dev-docs updates.

### Standalone Features (Wall Loss)

- [Wall Loss Builders, Mixins, and Factory][plan-wall-loss-builders] — Status:
  In Progress (P1, #818)
  - Scope: Builder/factory parity for wall loss with validation and unit
    conversion.
- [Rectangular Wall Loss Strategy][plan-rect] — Status:
  In Progress (P2, #817)
  - Scope: Rectangular chamber wall loss strategy with mirrored tests and
    exports.
- [Charged/Electrostatic Wall Loss Strategy][plan-charged-wall-loss] — Status:
  In Progress (P1, #821)
  - Scope: Charged wall loss with image-charge, optional E-field drift,
    builder/factory integration, neutral reduction path, docs/examples.
- [WallLoss Runnable Process][plan-wall-loss-runnable] — Status:
  Completed (P1, #819)
  - Scope: Runnable wrapping wall loss strategies with sub-step splitting,
    non-negative clamp, exports, and runnable-level tests.

## Maintenance Plans

- [Add Charge Support to add_concentration][plan-charge-add-concentration] —
  Status: Shipped (P2)
  - Scope: Enable `add_concentration()` to accept optional charge parameter for
    ion injection in coagulation simulations.

## References

- Parent wall loss epic: [#72](https://github.com/uncscode/particula/issues/72)

<!-- Epic Links -->
[epic-e1]: epics/E1-staggered-condensation-stepping.md

<!-- E1 Feature Links -->
[e1-f1]: features/E1-F1-core-staggered-stepping.md
[e1-f2]: features/E1-F2-batch-stepping-mode.md
[e1-f3]: features/E1-F3-builder-factory-integration.md
[e1-f4]: features/E1-F4-mass-conservation-validation.md
[e1-f5]: features/E1-F5-stability-performance-benchmarks.md
[e1-f6]: features/E1-F6-documentation-examples.md

<!-- Standalone Feature Links -->
[plan-wall-loss-builders]: features/wall-loss-builders-factory.md
[plan-rect]: features/rectangular-wall-loss-strategy.md
[plan-charged-wall-loss]: features/charged-wall-loss-strategy.md
[plan-wall-loss-runnable]: features/wall-loss-runnable.md

<!-- Maintenance Links -->
[plan-charge-add-concentration]: maintenance/M1-add-concentration-charge-support.md
