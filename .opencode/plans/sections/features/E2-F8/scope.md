# E2-F8 Scope

## In Scope

- Audit CPU condensation and coagulation data-container paths that accept
  `ParticleData` and/or `GasData`.
- Add or strengthen tests that distinguish:
  - data-container shape support (`n_boxes >= 1`), and
  - CPU strategy execution support (`n_boxes=1` for the covered CPU paths).
- Audit current public error-message behavior in P1, then tighten unsupported
  multi-box CPU coagulation execution to explicit rejection in P2.
- Update user-facing docs in `docs/Features/particle-data-migration.md` and,
  if helpful, `docs/Features/Roadmap/data-oriented-gpu.md`, so the final P3
  work remains docs-focused with no behavior change.
- Keep unit tests co-located with the dynamics modules they exercise.

## Out of Scope

- Implementing full CPU multi-box condensation or coagulation loops.
- Changing `ParticleData` or `GasData` schema shapes established by E2-F1.
- Introducing GPU kernels or changing dtype/mass schema decisions from E2-F6 or
  E2-F7.
- Redesigning public dynamics APIs beyond clearer validation and documentation.
- Changing runtime behavior in P3; the final phase should only publish the
  support contract already established by P1/P2 unless a wording-only mismatch
  is unavoidable.
- Creating a standalone testing-only phase; tests ship with the phase that
  changes behavior or documentation.

## Assumptions

- E2-F1 container shape conventions are the dependency baseline.
- Condensation's current `_require_single_box` behavior is intentional and
  should remain unless future tracks implement multi-box strategy execution.
- Coagulation `ParticleData` strategy calls should reject `n_boxes != 1` on CPU
  unless a future track implements broader execution semantics.

## Done Boundary

The feature is done when docs and tests clearly distinguish container multi-box
shape support from strategy-level single-box CPU execution support for the
covered dynamics paths, with the migration guide serving as the canonical
support contract and no P3 behavior changes.
