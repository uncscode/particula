# Phase Details

- [ ] **E2-F2-P1:** Define EnvironmentData fields and validation with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Add the core per-box field schema and strict validation rules for
    `temperature`, `pressure`, and canonical species-resolved
    `saturation_ratio` state.
  - Files: `particula/gas/environment_data.py`,
    `particula/gas/tests/environment_data_test.py`
  - Tests: valid single-box construction, invalid dimensionality, mismatched
    field lengths, non-finite values, negative pressure or saturation ratio,
    permitted supersaturation above `1.0`, and positive Kelvin temperature
    requirements.

- [ ] **E2-F2-P2:** Implement CPU dataclass exports and copy semantics with tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Complete the CPU container API with `n_boxes`, deep `copy()`, package
    exports, and multi-box behavior compatible with sibling containers.
  - Files: `particula/gas/environment_data.py`, `particula/gas/__init__.py`,
    `particula/gas/tests/environment_data_test.py`
  - Tests: multi-box valid construction, dtype coercion to `np.float64`,
    `n_boxes` property, copy independence, and import/export smoke tests.

- [ ] **E2-F2-P3:** Document process environment-state read and mutation boundaries
  - Issue: TBD | Size: XS | Status: Not Started
  - Goal: Update development docs so they explain that `EnvironmentData` owns
    per-box thermodynamic state, excluding simulation volume, while current
    process APIs remain scalar until downstream migration tracks.
  - Files: `docs/Features/particle-data-migration.md`,
    `docs/Features/Roadmap/data-oriented-gpu.md`
  - Tests: documentation link/reference validation where available; no
    standalone test phase is created.

## Phase Ordering Notes

- P1 must complete before P2 so the exported dataclass API matches the validated
  field list rather than a provisional schema.
- P2 is the implementation handoff for E2-F3 and E2-F5 because they depend on the
  final import path, `n_boxes`, and copy semantics.
- P3 should run after P2 so docs describe the shipped CPU contract and the scalar
  compatibility boundary that later phases still need to migrate.
