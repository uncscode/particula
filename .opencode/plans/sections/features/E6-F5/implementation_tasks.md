# Implementation Tasks

## CPU Core

- [ ] Add a shared documented active/free/invalid truth table in
  `particula/particles/slot_management.py`.
- [ ] Validate all particle fields as finite and nonnegative where required;
  reject positive mass with zero concentration, positive concentration with
  zero total mass, and nonzero charge in a free slot.
- [ ] Return ascending free indices and exact `np.int32` active/free counts.
- [ ] Define fixed-shape mass/concentration/charge request arrays with per-box
  valid-prefix counts and validate the complete request before writing.
- [ ] Activate request rank `r` into free rank `r`, preserving every shape,
  object identity, and unselected value.

## GPU Core

- [ ] Add private read-only classification/status kernels and deterministic
  box-local free-index enumeration in `particula/gpu/kernels/slot_management.py`.
- [ ] Validate particle/request arrays and every diagnostic sidecar for shape,
  `wp.float64`/`wp.int32` dtype, active device, and non-aliasing as applicable.
- [ ] Check per-box requested counts against free counts before clearing outputs
  or launching an activation write.
- [ ] Populate caller-owned active/free/activated counts exactly and preserve
  supplied sidecar identities.
- [ ] Keep CPU/Warp transfers explicit and avoid host-side index extraction in
  the successful direct GPU path.

## Tooling / Tests

- [ ] Add CPU truth-table, ordering, activation, identity, and atomic-failure
  tests in `particula/particles/tests/slot_management_test.py`.
- [ ] Add Warp CPU discovery, activation, parity, sidecar, and preflight tests in
  `particula/gpu/kernels/tests/slot_management_test.py`.
- [ ] Snapshot every caller-owned array for invalid-call tests and assert exact
  equality plus object identity afterward.
- [ ] Cover zero boxes/slots as supported by container conventions or document
  a preflight rejection consistently on CPU and GPU.
- [ ] Run optional CUDA tests with clean skips and retain existing coagulation,
  condensation, conversion, and particle-data regressions.

## Documentation

- [ ] Publish predicate and shape tables, ownership rules, direct import path,
  focused commands, and E6-F6/F7/F8 dependency boundaries.
