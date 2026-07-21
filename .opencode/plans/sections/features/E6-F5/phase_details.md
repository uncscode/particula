# Phase Details

- [ ] **E6-F5-P1:** Define CPU slot predicates and exact diagnostics with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Freeze active, free, and invalid-state semantics and return exact per-box counts plus ascending free indices without mutation.
  - Files: `particula/particles/slot_management.py`, `particula/particles/tests/slot_management_test.py`, `particula/particles/__init__.py`
  - Tests: Active/free truth table, contradictory states, sparse multi-box ordering, zero slots, exact integer counts, and source immutability.

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
