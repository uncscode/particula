# Implementation Tasks

## Backend

- [ ] Extend `ManifestEntry`/`ResourceManifest` in
  `particula/execution/gpu_resources.py` with enough immutable metadata to
  enumerate capture-required roles and capacity sources deterministically.
- [ ] Add checked element-count and logical-byte helpers shared by allocation
  validation and accounting; preserve zero-dimension semantics.
- [ ] Add frozen per-role, per-family, and aggregate byte report records that
  contain no pointers or payload data.
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

- [ ] Extend `particula/execution/tests/gpu_resources_test.py` with manifest,
  byte-formula, overflow, transactional publication, identity, and no-allocation
  tests for each phase.
- [ ] Add focused prepared-timestep integration tests under
  `particula/execution/tests/` with allocator/acquisition/readback spies.
- [ ] Cover one/multi-box, zero particle/species dimensions, both communication
  modes, absent communication, diagnostics selections, and capacity boundaries.
- [ ] Assert repeated compatible preparation returns the exact outer view,
  native records, arrays, capacities, and byte report.
- [ ] Run focused coverage-disabled assertions first, then the untargeted
  repository runner for full-package coverage; run strict MkDocs validation
  when documentation changes.
