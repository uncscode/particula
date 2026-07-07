# E2-F8 Scope

## In Scope

- Audit CPU condensation and coagulation data-container paths that accept
  `ParticleData` and/or `GasData`.
- Add or strengthen tests that distinguish:
  - data-container shape support (`n_boxes >= 1`), and
  - CPU strategy execution support (`n_boxes=1` or transitional box-0 only).
- Audit current public error-message and box-0-only behavior for unsupported
  multi-box strategy calls without changing runtime semantics in P1.
- Update user-facing docs in `docs/Features/particle-data-migration.md` and,
  if helpful, `docs/Features/Roadmap/data-oriented-gpu.md`.
- Keep unit tests co-located with the dynamics modules they exercise.

## Out of Scope

- Implementing full CPU multi-box condensation or coagulation loops.
- Changing `ParticleData` or `GasData` schema shapes established by E2-F1.
- Introducing GPU kernels or changing dtype/mass schema decisions from E2-F6 or
  E2-F7.
- Redesigning public dynamics APIs beyond clearer validation and documentation.
- Creating a standalone testing-only phase; tests ship with the phase that
  changes behavior or documentation.

## Assumptions

- E2-F1 container shape conventions are the dependency baseline.
- Condensation's current `_require_single_box` behavior is intentional and
  should remain unless future tracks implement multi-box strategy execution.
- Coagulation's current box-0 behavior should be either explicitly rejected for
  `n_boxes != 1` in a later phase or documented as transitional with tests
  proving boxes beyond zero are not executed.

## Done Boundary

The feature is done when docs and tests clearly distinguish container multi-box
shape support from strategy-level multi-box execution support for CPU dynamics.
