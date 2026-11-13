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
- [x] Require complete published `CaptureResourceRequirements` before final
  request construction and validate the cached publication at CAPTURED and READY
  admission without resource work.
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
- [x] Freeze the exact requirements/set/report triple in prepared timestep and
  simulation carriers and in the existing `configurations` signature group.
- [x] Preserve checkpoint resource enumeration and restart compatibility; this
  integration adds no continuation resources.

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
- [x] Add cached-validation, signature, prepared-carrier, real-loop, and
  documentation regressions, including prohibited-work spies and deterministic
  report snapshots.
