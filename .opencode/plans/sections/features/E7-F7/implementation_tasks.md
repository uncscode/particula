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
- [x] Pin one complete closed-map GAS or PARTICLES resource family, including
   map arrays, native buffers, and optional final volumes, by identity in
   `particula/execution/gpu_resources.py` (#1511).
- [x] Add schema-v2 optional communication checkpoint metadata/payload handling
   with fresh-identity restart and retained schema-v1 noncommunication restart
   compatibility (#1511).
- [x] Add typed `communication` and `volume_evolution` E7-F5 barriers and
   canonicalize the closed twelve-node schedule independent of registration
   order (#1511).
- [x] Invalidate only saturation ratio after either barrier; retain existing
   vapor-pressure and consumer refresh windows (#1511).

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
- [x] Preserve container/array identities and fixed dimensions across resident
   calls; reuse pinned session scratch without normal-step allocations (#1511).
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
- [x] Add resident resource, graph, executor, scheduler, checkpoint, and thermal
   update tests for canonical order, resource reuse, no hidden transfer/sync,
   schema-v1/v2 restart, and fault transitions (#1511).
- [ ] Add independent NumPy `float64` oracles in
  `particula/gpu/tests/communication_parity_test.py` without importing private
  production arithmetic.
- [ ] Cover one-box equivalence, isolated-box metamorphism, 1D chains, arbitrary
  pairs, sparse slots, expansion/compression, repeated steps, and explicit
  boundary ledgers on Warp CPU.
- [ ] Add optional CUDA rows using existing markers and clean skip conventions.
- [ ] Maintain at least 80% coverage for every changed module and do not lower
  repository thresholds.
