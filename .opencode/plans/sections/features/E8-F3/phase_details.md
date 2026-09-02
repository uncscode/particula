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

- [x] **E8-F3-P3:** Communication and diagnostic resource pinning with unit tests
  - Issue: #1563 | Size: S | Status: Shipped
  - Delivered: `GPUResourceRegistry` registers one absent or exact
    already-published closed GAS/PARTICLES communication view and ordered
    diagnostic registrations. It retains exact references plus deterministic
    schemas/logical-byte reports, validates transactionally on the host, and
    detects forbidden overlaps with an O(R log R) interval sweep.
  - Files: `particula/execution/gpu_resources.py`,
    `particula/execution/tests/gpu_resources_test.py`, adjacent diagnostics and
    resident-communication tests
  - Boundaries: No device I/O, allocation, public export, checkpoint-enumeration,
    or scheduler-behavior change; exact repeats reuse the retained inventory.

- [x] **E8-F3-P4:** Atomic capture-set preparation and exact identity reuse with unit tests
  - Issue: #1564 | Size: S | Status: Shipped
  - Delivered: Added direct-module-only `CaptureResourceRequirements` and
    `CaptureResourceSet`, atomic nonpublishing staging, metadata-only exact-set
    validation, and final publication only after candidate validation succeeds.
    Compatible repeats retain the original set/views by identity; candidate
    failure preserves existing publications and permits retry. Only new resident
    RNG stream sidecars initialize before publication.
  - Files: `particula/execution/gpu_resources.py`, narrowed optional
    resident-enqueue integration seam,
    `particula/execution/tests/gpu_resources_test.py`
  - Boundaries: The optional enqueue reference only retains/validates exact P4
    identities. It adds no absent-set error, READY/capture admission, token, or
    dispatch behavior; public exports and checkpoint enumeration remain unchanged.

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
