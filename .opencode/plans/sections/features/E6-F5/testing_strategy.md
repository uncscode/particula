# Testing Strategy

Coverage thresholds are not lowered. Every implementation phase ships its tests
in module-level `tests/` directories using the `*_test.py` suffix.

## Per-Phase Coverage

- **P1 (shipped, issue #1416):**
  `particula/particles/tests/slot_management_test.py` covers active/free and
  contradictory truth-table states, exact invalid-state errors, zero-species
  and zero-slot cases, sparse multi-box ascending indices with `-1` tails,
  `np.int32` counts, package export identity, and success/error-path source
  non-mutation with fresh-output allocation.
- **P2 (shipped, issue #1417):**
  `particula/particles/tests/slot_management_test.py` covers zero, empty, and
  zero-slot no-op boundaries; sparse multi-box/multi-species and exact-capacity
  mappings; fresh `np.int32` counts; request/destination identity and untouched
  storage; and atomic failures for malformed data/schema, selected-prefix
  values, aliasing, invalid existing state, and later-box capacity exhaustion.
- **P3 (shipped, issue #1418):**
  `particula/gpu/kernels/tests/slot_management_test.py` covers direct-Warp
  CPU-oracle discovery parity for sparse multi-box/multi-species, all-free,
  all-active, zero-box, zero-particle, and zero-species states; ascending
  `-1`-tailed indices; exact `int32` counts and sidecar identity; stale-output
  overwrite; malformed dtype/rank/shape/device and invalid-state rejection
  before output writes; density/volume non-access; and Warp CPU plus clean
  optional CUDA execution.
- **P4 (shipped, issue #1419):**
  `particula/gpu/kernels/tests/slot_management_test.py` compares particles and
  all four caller-owned `int32` sidecars exactly with independent CPU activation
  and post-call diagnostics. It covers ascending mapping, selected-prefix-only
  validation, zero prefixes/boxes/capacity, exact and sparse capacity, repeated
  activation, package export identity, and optional CUDA clean skips. Snapshot
  tests prove non-mutation for schema, state, count, selected-record, capacity,
  and direct/partial alias rejection.
- **P5:** Documentation link/import/shape-table validation and focused command
  review.

## Test Locations

- `particula/particles/tests/slot_management_test.py`
- `particula/gpu/kernels/tests/slot_management_test.py`
- Existing regression suites under `particula/particles/tests/`,
  `particula/gpu/tests/`, and `particula/gpu/kernels/tests/`.

## Parity and Precision

Predicates, indices, and diagnostics require exact equality, not tolerances.
Activated `float64` fields copy source values exactly. Tests compare arrays,
shapes, dtypes, devices, and identities separately. Warp CPU is mandatory when
Warp is installed; CUDA uses the repository's clean optional skip policy.

## Coverage Impact

All public and private helpers receive branch coverage for zero work, sparse
state, full state, invalid state, insufficient capacity, and multi-box inputs.
Changed-code coverage must remain at least 80%, with no exclusion or threshold
changes.
