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

- [ ] **E6-F5-P2:** Implement deterministic CPU slot activation with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Copy fixed-shape request records into ascending free slots only after full validation and preserve container shapes and identities.
  - Files: `particula/particles/slot_management.py`, `particula/particles/tests/slot_management_test.py`
  - Tests: Zero, partial, and exact-capacity requests; multi-species/box mapping; activated counts; untouched slots; capacity and malformed-request failure atomicity.

- [ ] **E6-F5-P3:** Implement GPU slot discovery and diagnostics with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Populate caller-owned active/free counts and fixed-shape ascending free indices on the active device with no particle mutation.
  - Files: `particula/gpu/kernels/slot_management.py`, `particula/gpu/kernels/tests/slot_management_test.py`
  - Tests: Warp CPU predicate/count/index parity, sidecar shape/dtype/device checks, sentinel contents, supplied identity, invalid-state rejection, and optional CUDA execution.

- [ ] **E6-F5-P4:** Implement atomic GPU slot activation and CPU parity tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Validate state, requests, capacity, and sidecars before launching deterministic activation and exact diagnostic updates.
  - Files: `particula/gpu/kernels/slot_management.py`, `particula/gpu/kernels/tests/slot_management_test.py`, `particula/gpu/kernels/__init__.py`
  - Tests: CPU/Warp mapping and value parity, zero/exact-capacity cases, fixed identities/shapes, diagnostic equality, no mutation on every invalid input, and CUDA clean skips.

- [ ] **E6-F5-P5:** Update development documentation
  - Issue: TBD | Size: XS | Status: Not Started
  - Goal: Document predicates, fixed-shape request/diagnostic contracts, validation order, direct imports, downstream boundaries, and focused test commands.
  - Files: `AGENTS.md`, `docs/Features/`, `.opencode/guides/`, E6 plan sections as needed
  - Tests: Markdown links, import snippets, shape tables, terminology, and supported/deferred boundary review.
