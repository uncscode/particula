# Implementation Tasks

## Backend

- [x] Add a separate immutable declaration-metadata layer and direct-module-only
  inventory carriers in `particula/execution/gpu_resources.py`; existing
  `ManifestEntry`/`ResourceManifest` declarations and ordering remain unchanged.
- [x] Add checked shape, stride, element-count, and logical-byte helpers shared
  by allocation, range validation, and reporting, preserving zero dimensions.
- [x] Add frozen pointer-free per-role, per-family, and aggregate logical-byte
  report records plus the read-only `logical_resource_report()` accessor.
- [x] Add the descriptor-only dilution normalized-coefficient and factors roles
  and concrete `PreparedResourceViews` plumbing; validate supplied views
  read-only and retain accepted resource identities in prepared adapters.
- [ ] Inventory the remaining E8-F2 prepared operations and add missing
  fixed-shape process, control, selected-lane, and validation/status sidecars
  to canonical manifests.
- [x] Register ordered diagnostic outputs/accounting work as concrete-only
  selected capture resources while preserving direct diagnostics validation.
- [x] Preserve mutually exclusive GAS/PARTICLES communication families and pin
  one selected already-published view, including configuration, maps, work
  records, status, snapshots, and optional final volumes, by identity.
- [x] Implement whole-set preflight, nonpublishing candidate allocation,
  cross-family nonalias validation, newly-created RNG initialization, and atomic
  publication in `particula/execution/gpu_resources.py`.
- [x] Add exact capture-set validation/accessors that cannot allocate, acquire,
  reseed, transfer, synchronize, inspect payloads, or mutate bindings.
- [x] Add narrowed optional E8-F2 prepared-record retention/validation of an
  exact capture set and prepared views; READY/capture admission remains P5.
- [ ] Preserve checkpoint resource enumeration and restart compatibility; add
  new continuation resources only when their semantics require checkpointing.

## Tooling / Tests

- [x] Extend `particula/execution/tests/gpu_resources_test.py` with P1 manifest
  order, independent byte-formula, zero-extent, invalid input, overflow,
  immutability, export-boundary, and read-only reporting tests.
- [x] Add adjacent P2 prepared-view validation and exact identity-retention
  coverage without introducing allocation/publication/reacquisition behavior.
- [x] Add focused registry/diagnostics/communication tests for registration,
  report reuse, host-only metadata behavior, and transactional rejection.
- [x] Cover both communication modes, absent communication, ordered diagnostic
  selections, zero extents, exact identities, and overlap boundaries.
- [x] Assert repeated compatible preparation returns the exact outer view,
  native records, arrays, capacities, and byte report, and that distinct but
  value-equal requirements reject.
- [ ] Run focused coverage-disabled assertions first, then the untargeted
  repository runner for full-package coverage; run strict MkDocs validation
  when documentation changes.
