# Implementation Tasks

## Scientific Model and Backend

- [x] Create `particula/dynamics/nucleation/` with typed immutable strategy,
  validity-domain, injection-composition, and formation-metadata APIs.
- [x] Implement overflow-safe SI conversion and activation/kinetic potential
  rates without hidden clipping or out-of-range extrapolation.
- [x] Implement a pure P2 finalizer for event demand, per-event species mass,
  shared gas admission, deterministic limiting diagnostics, and bounded
  rounding correction without mutation. E6-F5 requests remain future work.
- [x] Integrate E6-F5 activation and E6-F6 resampling/scaling on private P3
  staging; require complete all-box validation before particle or gas writes.
- [x] Package final represented demand into equal-weight slots without residual
  truncation; preserve final represented per-species mass.
- [x] Add strict atomic builders/factory with explicit units, domain, precursor
   index, composition, formation size, survival factor, and provenance.
- [x] Add public CPU-only single-box `Nucleation` and immutable
  `NucleationCommitConfig` to `particle_process.py`; recompute current gas rate
  and inventory on every equal substep while retaining backing-container identity.
- [x] Export only intended P4 construction APIs through nucleation and dynamics
   initializers; retain P2/P3 names as concrete-module-only.

## Tooling and Tests

- [x] Add isolated equation/domain/order/validation tests in
  `nucleation_strategies_test.py`.
- [x] Add source-record tests in `particle_source_test.py` for inventory
  admission, diagnostics, immutable ownership, validation, and nonmutation.
- [x] Add P3 capacity, no-op, scaling, atomic-rejection, immutable-record, and
  export-boundary tests in `particle_source_test.py`.
- [x] Add builder/factory tests for units, required fields, strict schemas,
   atomic recovery, factory isolation, and invalid aliases.
- [x] Add runnable tests in `nucleation_runnable_test.py` for topology,
  validation, identity, sequencing, no-ops, composability, and failure
  boundaries.
- [x] Add independent, test-local NumPy `float64` P2/P3 oracles and snapshots
  in `particle_source_test.py` and
  `particula/integration_tests/nucleation_process_test.py`; do not derive
  expected values with production helpers.
- [x] Add multi-box/multi-species conservation, capacity-policy,
  failure-atomicity, repeated-current-gas, and limiting-species coverage, plus
  self-contained single-box P5 gas-coupling and identity regressions in
  `nucleation_runnable_test.py`.
- [ ] Run focused tests, full fast pytest, Ruff, and mypy without reducing
  coverage thresholds.

## Documentation

- [ ] Update theory with the shipped API, units, domains, citations, and model
  boundary.
- [ ] Add a supported CPU example and cross-link E6-F5/F6 and E6-F8/F9.
- [ ] Record focused commands and conservation tolerances in `AGENTS.md` and
  the user-facing feature document.
