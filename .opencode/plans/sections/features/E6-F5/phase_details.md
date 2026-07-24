# Phase Details

## Sequencing

Complete the CPU predicate and activation phases P1/P2 before GPU discovery in
P3 and GPU activation/parity in P4; P5 documents the completed contract.

- [x] **E6-F5-P1:** Define CPU slot predicates and exact diagnostics with unit tests
  - Issue: #1416 | Size: S | Status: Shipped
  - Delivered: `get_slot_diagnostics(data)` freezes the read-only CPU active,
    free, and invalid-state contract. It returns fresh fixed-shape `np.int32`
    free-index, active-count, and free-count sidecars; free rows are ascending
    with `-1` tails. Invalid state raises exactly
    `ValueError("Invalid particle slot state.")`.
  - Files: `particula/particles/slot_management.py`,
    `particula/particles/tests/slot_management_test.py`,
    `particula/particles/__init__.py`
  - Tests: Truth-table and contradictory-state coverage, zero-species/zero-slot
    cases, sparse multi-box ordering, exact integer diagnostics, public export,
    and success/error-path source non-mutation plus fresh-allocation checks.

- [x] **E6-F5-P2:** Implement deterministic CPU slot activation with unit tests
  - Issue: #1417 | Size: S | Status: Shipped
  - Delivered: direct-import `activate_slots` maps each declared request prefix
    to ascending free slots from `get_slot_diagnostics`, mutates only selected
    mass/concentration/charge records, and returns fresh per-box `np.int32`
    activated counts. Complete read-only preflight validates `ParticleData`,
    writable destination schema, request schema and overlap, selected values,
    canonical existing slot state, and capacity before any write.
  - Files: `particula/particles/slot_management.py`, `particula/particles/tests/slot_management_test.py`
  - Tests: Zero/empty/zero-slot, sparse multi-box multi-species, and exact-free
    capacity success cases; identity and untouched-storage checks; and atomic
    rejection of malformed data/requests, invalid selected prefixes, aliases,
    invalid existing slots, and later-box capacity failures.

- [x] **E6-F5-P3:** Implement GPU slot discovery and diagnostics with unit tests
  - Issue: #1418 | Size: S | Status: Shipped
  - Delivered: concrete, direct-import-only `get_slot_diagnostics_gpu` performs
    read-only Warp classification from mass, concentration, and charge; it
    returns supplied same-device `wp.int32` free-index, active-count, and
    free-count sidecars by identity. Free indices are ascending with `-1` tails.
    Invalid schema or slot state raises before a writer launch, so particle
    fields and stale output sidecars remain unchanged. Density and volume are
    not read or validated, and the function has no package export.
  - Files: `particula/gpu/kernels/slot_management.py`, `particula/gpu/kernels/tests/slot_management_test.py`
  - Tests: direct-Warp CPU-oracle predicate/count/index parity; sidecar
    shape/dtype/device and identity checks; stale-output overwrite; invalid
    state and malformed-schema atomicity; density/volume non-access; and
    optional CUDA execution with clean skips.

- [x] **E6-F5-P4:** Implement atomic GPU slot activation and CPU parity tests
  - Issue: #1419 | Size: S | Status: Shipped
  - Delivered: package-exported `activate_slots_gpu` validates same-device
    schemas, ownership/aliasing, canonical slot state, selected counts,
    capacity, and selected request records before writes. It maps each selected
    request-prefix rank to the ascending free fixed slot, mutates only mass,
    concentration, and charge, and returns caller-owned `wp.int32`
    `(activated_counts, free_indices, active_counts, free_counts)` sidecars by
    identity. P3 `get_slot_diagnostics_gpu` remains concrete-module-only.
  - Files: `particula/gpu/kernels/slot_management.py`,
    `particula/gpu/kernels/tests/slot_management_test.py`,
    `particula/gpu/kernels/__init__.py`
  - Tests: independent CPU-oracle particle/output parity; ascending mapping;
    zero-prefix, zero-box, zero-capacity, exact-capacity, repeated, and sparse
    capacity cases; schema/state/count/request/capacity/alias rejection
    atomicity; package exports; and optional CUDA clean skips.

- [x] **E6-F5-P5:** Update development documentation
  - Issue: #1420 | Size: XS | Status: Shipped
  - Delivered: documented the shared predicate, fixed-shape CPU/direct-Warp
    schemas, imports, return ownership, preflight boundaries, focused commands,
    and E6-F6/F7/F8/F9 ownership without adding a user example.
  - Files: `AGENTS.md`, `docs/Features/`, `.opencode/guides/`, E6 plan sections
  - Tests: Markdown links, import snippets, shape tables, terminology, and
    supported/deferred boundary review.
