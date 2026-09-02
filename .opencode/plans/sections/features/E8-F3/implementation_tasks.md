# Implementation Tasks

## Backend

- [x] Add a separate immutable declaration-metadata layer and direct-module-only
  inventory carriers in `particula/execution/gpu_resources.py`; existing
  `ManifestEntry`/`ResourceManifest` declarations and ordering remain unchanged.
- [x] Add checked shape, stride, element-count, and logical-byte helpers shared
  by allocation, range validation, and reporting, preserving zero dimensions.
- [x] Add frozen pointer-free per-role, per-family, and aggregate logical-byte
  report records plus the read-only `logical_resource_report()` accessor.
- [ ] Inventory E8-F2 prepared operations and add missing fixed-shape process,
  control, selected-lane, and validation/status sidecars to canonical manifests.
- [ ] Represent diagnostic outputs/accounting work as closed capture resources
  while preserving existing direct diagnostics validation behavior.
- [ ] Preserve mutually exclusive GAS/PARTICLES communication families and pin
  configuration, map arrays, work records, status, snapshots, and optional
  final volumes by identity.
- [ ] Implement whole-set preflight, candidate allocation, cross-family
  nonalias validation, explicit RNG initialization, and atomic publication.
- [ ] Add exact capture-set validation and accessors that cannot allocate,
  acquire, reseed, transfer, synchronize, inspect payloads, or mutate bindings.
- [ ] Integrate the exact capture resource set with E8-F2 prepared records and
  E8-F1 compatibility/READY checks.
- [ ] Preserve checkpoint resource enumeration and restart compatibility; add
  new continuation resources only when their semantics require checkpointing.

## Tooling / Tests

- [x] Extend `particula/execution/tests/gpu_resources_test.py` with P1 manifest
  order, independent byte-formula, zero-extent, invalid input, overflow,
  immutability, export-boundary, and read-only reporting tests.
- [ ] Add focused prepared-timestep integration tests under
  `particula/execution/tests/` with allocator/acquisition/readback spies.
- [ ] Cover one/multi-box, zero particle/species dimensions, both communication
  modes, absent communication, diagnostics selections, and capacity boundaries.
- [ ] Assert repeated compatible preparation returns the exact outer view,
  native records, arrays, capacities, and byte report.
- [ ] Run focused coverage-disabled assertions first, then the untargeted
  repository runner for full-package coverage; run strict MkDocs validation
  when documentation changes.
