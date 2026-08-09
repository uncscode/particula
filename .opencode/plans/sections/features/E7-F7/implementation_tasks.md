# Implementation Tasks

## Execution Layer

- [x] Add immutable communication-map, transport-mode, resource-shape, optional
  volume-update, and configuration declarations in
  `particula/execution/communication.py` (E7-F7-P1, #1507). The module is
  concrete-only and intentionally unexported.
- [x] Implement deterministic, read-only validation of map schemas, dimensions,
  device metadata, aliases, enabled-edge topology, duplicate directed edges,
  rates, and optional final volumes (E7-F7-P1, #1507). Population-dependent
  outbound overdraw is deferred to P3.
- [ ] Extend E7-F4 resource registration with fixed-shape communication scratch,
  status, slot-plan, and diagnostic buffers.
- [ ] Extend E7-F4 checkpoint metadata/resources so restart preserves configured
  map identity and any documented mutable communication state.
- [ ] Add typed E7-F5 node kinds for communication and volume evolution and place
  them at the canonical pre-process barrier independent of registration order.
- [ ] Mark gas-derived fields stale after communication and route refresh through
  existing E7-F5 update dependencies rather than refreshing inside transport.
- [ ] Add E7-F6 capability reasons for unsupported representation/map/device
  combinations; preserve default error policy and pre-selection-only fallback.

## GPU Kernels

- [x] Add `particula/gpu/kernels/communication.py` with the direct,
  device-resident P2 volume-normalization operation (E7-F7-P2, #1508).
- [x] Validate P2 caller-owned final volumes and particle/gas primary storage
  for schemas, devices, aliases, finite physical domains, and safe scaling
  before its apply writer launches (E7-F7-P2, #1508).
- [x] Add staged amount-ledger gas-transfer kernels (E7-F7-P3, #1509); particle
  planning/commit remains P4.
- [x] Validate P3 aggregate outbound sums before its single gas-state commit
  (E7-F7-P3, #1509); P4 destination capacity remains deferred.
- [x] Keep P3 source reads synchronous by staging immutable pre-node amounts;
  transfers are registration-order independent (E7-F7-P3, #1509).
- [ ] Preserve container/array identities and fixed dimensions across successful
  calls; reuse session scratch rather than allocating in repeated steps.
- [x] Implement P3 closed-map ledger conservation and explicit open-boundary
  source/sink accounting without host readback in the normal path (#1509).
- [x] Preserve per-particle species mass and signed charge during P4 slot
  transport and use immutable pre-step deterministic exact matching or
  free-slot selection without resize or compaction (#1510).
- [x] Ensure unchanged P2 volumes are write-free after complete validation
  (E7-F7-P2, #1508).

## Tooling / Tests

- [x] Add `particula/execution/tests/communication_test.py` for P1 declaration
  and validation contracts (E7-F7-P1, #1507).
- [x] Add `particula/gpu/kernels/tests/communication_test.py` for P2 volume,
  particle/gas inventory, rejection atomicity, identity, and no-op behavior
   (E7-F7-P2, #1508).
- [x] Extend the co-located module for P3 immutable-ledger, open-boundary,
   order-independence, no-op, and precommit-gating coverage (#1509).
- [x] Extend the co-located module for P4 fixed-capacity particle transport,
  immutable planning, deterministic assignments, closed-map conservation, and
  gated commit coverage (#1510).
- [ ] Add scheduler/session tests for canonical order, resource reuse, no hidden
  transfer/sync, checkpoint/restart, capability errors, and fault transitions.
- [ ] Add independent NumPy `float64` oracles in
  `particula/gpu/tests/communication_parity_test.py` without importing private
  production arithmetic.
- [ ] Cover one-box equivalence, isolated-box metamorphism, 1D chains, arbitrary
  pairs, sparse slots, expansion/compression, repeated steps, and explicit
  boundary ledgers on Warp CPU.
- [ ] Add optional CUDA rows using existing markers and clean skip conventions.
- [ ] Maintain at least 80% coverage for every changed module and do not lower
  repository thresholds.
