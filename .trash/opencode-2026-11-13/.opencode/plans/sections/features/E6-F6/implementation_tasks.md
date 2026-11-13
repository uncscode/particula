# Implementation Tasks

## CPU Core

- [x] Define immutable policy configuration and exact precedence resolver in
  `particula/particles/exhaustion.py`.
- [x] Define frozen fixed-shape planning contracts for already-discovered
  capacity sidecars; no E6-F5 discovery, activation, or E6-F7 source records
  are implemented in P1.
- [x] Implement read-only all-box validation and deferred-policy planning.
- [x] Add float64 tuple-backed weighted number, species-mass, and charge
  inventories with documented later-commit accounting.
- [x] Implement deterministic stable interval-sweep resampling selection,
  conservation/moment accounting, and one all-box-preflighted commit boundary
  without resizing.
- [x] Implement bounded same-direction representative-volume, raw-weight, and
  source-demand scaling as the concrete CPU helper
  `apply_representative_volume_scaling`.
- [x] Keep the P1 API concrete-only; no `particula.particles` re-export was
  added.

## Direct Warp

- [x] Add `resampling_step_gpu` and concrete-only `ResamplingBuffers` in
  `particula/gpu/kernels/exhaustion.py` for explicit release-count direct Warp
  remapping.
- [x] Implement allocation-stable read-only validation, device-resident staged
  bitonic-sort/interval-sweep planning, diagnostic gating, and a single commit.
- [x] Reject invalid schema, values, counts, bounds, devices, and aliased
  buffers before caller mutation; diagnostic planning failures skip commit.
- [x] Export only `resampling_step_gpu` through `particula.gpu.kernels`; keep
  `ResamplingBuffers` concrete-module-only.
- [x] Add concrete-only direct Warp representative-volume scaling with fused
  read-only preflight, bounded status gating, diagnostic output, and a
  selected-row-only commit; P1 policy resolution remains deferred.

## Tooling / Tests

- [x] Add `particula/particles/tests/exhaustion_test.py` covering strict frozen
  controls and records, all-box validation/no-mutation, policy precedence,
  empty dimensions, and an independent float64 weighted-inventory oracle.
- [x] Extend the co-located CPU suite with deterministic detached P2 plans,
  fixed-capacity clearing, independent equal-stratum remap checks, diagnostic
  bound validation, and stale/later-box malformed-plan atomicity.
- [x] Add `particula/gpu/kernels/tests/exhaustion_test.py` for Warp CPU parity,
  optional CUDA, supplied-buffer ownership, diagnostics, and invalid-call
  snapshots.
- [x] Extend CPU/Warp exhaustion and kernel-export tests for P4 schema/value/
  atomicity, independent-oracle parity, no-selected diagnostics, empty boxes,
  optional CUDA, and concrete-only import surfaces.
- [x] Add test-only independent CPU/Warp conservation ledgers and admitted-source
  fixtures for P2 sparse/full/no-op/over-capacity cases and P4 selected/
  unselected representative-volume accounting.
- [ ] Run focused tests, full fast regressions, Ruff, mypy, and docs validation
  without reducing the repository coverage threshold.
