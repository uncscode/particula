# Development Plans

Feature and maintenance development plans tracked for particula. Each entry
follows the appropriate template to capture motivation, scope, phases, testing,
and rollout.

## Epics

- [E1: Staggered ODE Stepping for Particle-Resolved Condensation][epic-staggered-condensation]
  — Status: Not Started (P2)
  - Scope: Staggered ODE stepping framework for particle-resolved condensation
    with three modes (half-step, random, batch) for improved stability and
    mass conservation.

## Maintenance Plans

- [Add Charge Support to add_concentration][plan-charge-add-concentration] —
  Status: Shipped (P2)
  - Scope: Enable `add_concentration()` to accept optional charge parameter for
    ion injection in coagulation simulations.

## Feature Plans

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

## References

- Parent wall loss epic: [#72](https://github.com/uncscode/particula/issues/72)

[epic-staggered-condensation]: epics/E1-staggered-condensation-stepping.md
[plan-wall-loss-builders]: features/wall-loss-builders-factory.md
[plan-rect]: features/rectangular-wall-loss-strategy.md
[plan-charged-wall-loss]: features/charged-wall-loss-strategy.md
[plan-wall-loss-runnable]: features/wall-loss-runnable.md
[plan-charge-add-concentration]: maintenance/M1-add-concentration-charge-support.md
