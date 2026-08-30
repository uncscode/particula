# Phase Details

- [ ] **E8-F3-P1:** Canonical capture resource inventory and byte formulas with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Represent every capture-lifetime role with stable family/role order,
    shape, dtype, capacity source, ownership, and overflow-safe logical bytes.
  - Files: `particula/execution/gpu_resources.py`,
    `particula/execution/tests/gpu_resources_test.py`
  - Tests: Exact manifest order, formulas for normal/zero dimensions, dynamic
    collision/edge capacities, dtype sizes, checked overflow, and stable totals.

- [ ] **E8-F3-P2:** Complete process and control sidecar preallocation with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Add missing fixed-shape process, normalized-control, selected-lane,
    and validation/status storage required by E8-F2 without changing public
    direct-kernel APIs.
  - Files: `particula/execution/gpu_resources.py`, prepared process adapters,
    `particula/execution/tests/gpu_resources_test.py`
  - Tests: Complete records, device/schema/nonalias checks, zero-size forms,
    repeated identity, and no allocation after first publication.

- [ ] **E8-F3-P3:** Communication and diagnostic resource pinning with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Bind the selected closed communication family and complete diagnostic
    outputs/accounting work as capture-lifetime resources with exact identities.
  - Files: `particula/execution/gpu_resources.py`,
    `particula/execution/diagnostics.py`,
    `particula/execution/resident_communication.py`, adjacent tests
  - Tests: Mode-specific inventories, immutable map/configuration identity,
    diagnostic role uniqueness, alias rejection, and deterministic accounting.

- [ ] **E8-F3-P4:** Atomic capture-set preparation and exact identity reuse with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Preflight and stage the whole E8-F2 requirement set before atomic
    publication, then validate/retrieve that exact set without allocation.
  - Files: `particula/execution/gpu_resources.py`, E8-F2 preparation module,
    `particula/execution/tests/gpu_resources_test.py`
  - Tests: Mid-allocation failure leaves no capture set, retry succeeds,
    compatible reacquisition preserves every identity, and all drift/replacement
    forms reject before allocator or writer calls.

- [ ] **E8-F3-P5:** Prepared-timestep integration, accounting validation, and documentation
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Require the complete pinned set before READY/capture, prove zero
    allocator activity on repeated prepared execution, and document ownership
    and accounting boundaries.
  - Files: E8-F1/E8-F2 concrete execution modules,
    `particula/execution/tests/`, `docs/Features/Roadmap/data-oriented-gpu.md`,
    `.opencode/guides/testing_guide.md` if commands change
  - Tests: Prepared-path integration, forbidden-allocation spies, exact byte
    report snapshots, focused resident regression, and strict docs validation.
