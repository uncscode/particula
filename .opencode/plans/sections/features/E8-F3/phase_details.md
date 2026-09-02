# Phase Details

- [x] **E8-F3-P1:** Canonical capture resource inventory and byte formulas with unit tests
  - Issue: #1561 | Size: S | Status: Shipped
  - Delivered: Immutable direct-module-only reports resolve all six existing
    manifests in stable order, including both communication families, with
    shape, dtype, capacity source, ownership, element counts, and logical bytes.
    The accessor is read-only and independent of acquisition/configuration
    payloads; checked arithmetic is shared with allocation/range validation.
  - Files: `particula/execution/gpu_resources.py`,
    `particula/execution/tests/gpu_resources_test.py`
  - Tests: Focused passing coverage for manifest order, normal/zero shapes,
    dynamic collision/edge capacities, dtype sizes, checked overflow, stable
    totals, frozen carriers, direct-module-only exports, and no mutation before
    or after unrelated acquisition.

- [x] **E8-F3-P2:** Complete process and control sidecar preallocation with unit tests
  - Issue: #1562 | Size: S | Status: Shipped
  - Delivered: Added the descriptor-only dilution family with `(B,)`
    `wp.float64` normalized-coefficient and factors roles, concrete-only
    `PreparedResourceViews`, read-only supplied-view validation, and prepared
    adapter retention of supplied resource identities. No allocation,
    publication, reacquisition, RNG change, or public export was added.
  - Files: `particula/execution/gpu_resources.py`, prepared dilution/resource
    adapter seams, adjacent execution tests
  - Validation: Supplied prepared views are schema-validated read-only before
    adapter use; accepted resources retain exact identity and rejected views do
    not mutate supplied arrays.

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
