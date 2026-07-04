# Phase Details

- [x] **E2-F2-P1:** Define EnvironmentData fields and validation with unit tests
  - Issue: #1188 | Size: S | Status: Shipped
  - Goal: Add the core per-box field schema and strict validation rules for
    `temperature`, `pressure`, and canonical species-resolved
    `saturation_ratio` state.
  - Files: `particula/gas/environment_data.py`,
    `particula/gas/tests/environment_data_test.py`
  - Shipped details: constructor inputs are coerced with
    `np.asarray(..., dtype=np.float64)`, validation runs in deterministic stages
    (dimensionality, shared box-count shape, finiteness, then physical bounds),
    and direct-module import is covered without adding package exports.
  - Tests: valid single-box construction, valid multi-box construction,
    list/tuple dtype coercion, invalid dimensionality, mismatched field
    lengths, non-finite values, nonpositive pressure and negative
    saturation_ratio,
    permitted supersaturation above `1.0`, positive Kelvin temperature
    requirements, and helper-level validation smoke coverage.

- [x] **E2-F2-P2:** Implement CPU dataclass exports and copy semantics with tests
  - Issue: #1189 | Size: S | Status: Shipped
  - Goal: Complete the CPU container API with `n_boxes`, deep `copy()`, package
    exports, and multi-box behavior compatible with sibling containers.
  - Files: `particula/gas/environment_data.py`, `particula/gas/__init__.py`,
    `particula/gas/tests/environment_data_test.py`
  - Shipped details: `EnvironmentData.n_boxes` now derives box count from the
    validated temperature axis, `copy()` rebuilds the dataclass from
    independent NumPy arrays, and `particula.gas.__init__` exports
    `EnvironmentData` as the package-level import path without changing the P1
    constructor validation flow.
  - Tests: `environment_data_test.py` now covers `n_boxes`, copy memory
    independence, mutation isolation for copied temperature/pressure/
    `saturation_ratio` arrays, retained multi-box and dtype-coercion behavior,
    and package-export smoke coverage.

- [x] **E2-F2-P3:** Document process environment-state read and mutation boundaries
  - Issue: #1190 | Size: XS | Status: Shipped
  - Goal: Update development docs so they explain that `EnvironmentData` owns
    per-box thermodynamic state, excluding simulation volume, while current
    process APIs remain scalar until downstream migration tracks.
  - Files: `docs/Features/particle-data-migration.md`,
    `docs/Features/Roadmap/data-oriented-gpu.md`
  - Shipped details: both docs now describe the shipped CPU
    `EnvironmentData` contract consistently, naming `temperature`, `pressure`,
    and `saturation_ratio` as environment-owned state, keeping simulation
    volume under `ParticleData.volume`, preserving scalar `temperature` /
    `pressure` process APIs as the current compatibility boundary, and leaving
    GPU mirrors, conversion helpers, and runtime integration downstream.
  - Tests: documentation-only validation via changed-section review and
    reference/link inspection; no standalone test phase is created.

## Phase Ordering Notes

- P1 must complete before P2 so the exported dataclass API matches the validated
  field list rather than a provisional schema.
- P2 is now the implementation handoff for E2-F3 and E2-F5 because they can
  depend on the shipped import path, `n_boxes`, and copy semantics.
- P3 should run after P2 so docs describe the shipped CPU contract and the scalar
  compatibility boundary that later phases still need to migrate.
