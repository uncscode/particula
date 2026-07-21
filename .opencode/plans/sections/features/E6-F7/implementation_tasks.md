# Implementation Tasks

## Scientific Model and Backend

- [ ] Create `particula/dynamics/nucleation/` and typed strategy,
  validity-domain, injection-composition, source-record, and diagnostics APIs.
- [ ] Implement SI conversion and activation/kinetic rates without hidden
  clipping or out-of-range extrapolation.
- [ ] Implement a pure finalizer for event demand, per-event species mass,
  shared gas admission, and fixed-shape E6-F5 requests without mutation.
- [ ] Integrate E6-F5 activation and E6-F6 plan/commit; require a complete plan
  for every box before gas or particle writes.
- [ ] Ensure source packaging changes computational weight only, never
  represented event count or per-species mass.
- [ ] Add builders/factory with explicit units, domain, precursor index,
  composition, formation size, and survival factor.
- [ ] Add `Nucleation` to `particle_process.py`; recompute current gas rate and
  inventory on every substep.
- [ ] Export only intended APIs through nucleation and dynamics initializers.

## Tooling and Tests

- [ ] Add equation/domain tests in `nucleation_strategies_test.py`.
- [ ] Add source, capacity, atomicity, and conservation tests in
  `particle_source_test.py`.
- [ ] Add builder/factory tests for units, required fields, and invalid aliases.
- [ ] Add runnable tests in `nucleation_runnable_test.py`.
- [ ] Add an independent multi-box/species oracle under
  `particula/integration_tests/`; do not derive expected values with production.
- [ ] Run focused tests, full fast pytest, Ruff, and mypy without reducing
  coverage thresholds.

## Documentation

- [ ] Update theory with the shipped API, units, domains, citations, and model
  boundary.
- [ ] Add a supported CPU example and cross-link E6-F5/F6 and E6-F8/F9.
- [ ] Record focused commands and conservation tolerances in `AGENTS.md` and
  the user-facing feature document.
