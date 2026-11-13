# Implementation Tasks

## CPU Core

- [x] Add a shared documented active/free/invalid truth table in
  `particula/particles/slot_management.py`.
- [x] Validate all particle fields as finite and nonnegative where required;
  reject positive mass with zero concentration, positive concentration with
  zero total mass, and nonzero charge in a free slot.
- [x] Return ascending free indices and exact `np.int32` active/free counts.
- [x] Define fixed-shape mass/concentration/charge request arrays with per-box
  valid-prefix counts and validate the complete request before writing.
- [x] Activate request rank `r` into free rank `r`, preserving every shape,
  object identity, and unselected value.

## GPU Core

- [x] Add private read-only classification/status kernels and deterministic
  box-local free-index enumeration in `particula/gpu/kernels/slot_management.py`.
- [x] Validate P3 particle classification fields and diagnostic sidecars for
  shape, `wp.float64`/`wp.int32` dtype, and active device before output writes.
- [x] Check per-box requested counts against free counts before clearing outputs
  or launching an activation write.
- [x] Populate caller-owned active/free counts and free indices exactly, with
  `-1` tails, while preserving supplied sidecar identities.
- [x] Keep CPU/Warp transfers explicit and avoid host-side index extraction in
  the successful direct GPU path.
- [x] Add package-exported `activate_slots_gpu` with selected-prefix validation,
  ascending-free-slot mapping, and caller-owned activation/diagnostic sidecars.

## Tooling / Tests

- [x] Add CPU truth-table, ordering, identity, invalid-state non-mutation, and
  activation tests in `particula/particles/tests/slot_management_test.py`.
- [x] Add Warp CPU discovery, parity, sidecar, and preflight tests in
  `particula/gpu/kernels/tests/slot_management_test.py`.
- [x] Snapshot every caller-owned array for P1 invalid-call tests and assert exact
  equality plus object identity afterward.
- [x] Cover CPU zero-slot activation: zero counts are exact no-ops and positive
  requests fail atomically; P4 covers matching GPU zero-capacity behavior.
- [x] Run P3 optional CUDA rows with clean skips and direct-Warp discovery
  coverage.
- [x] Add P4 CPU-oracle parity, rejection atomicity, aliasing, sparse-capacity,
  zero-case, package-export, and optional CUDA activation coverage.

## Documentation

- [x] Publish P4 ownership, direct import, preflight, rollback, and focused-test
  guidance in `AGENTS.md`, `docs/index.md`, and architecture/testing guides.
