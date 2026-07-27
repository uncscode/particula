# Implementation Tasks

## Execution Layer

- [ ] Add immutable communication, edge-map, transport-mode, volume-update, and
  boundary-ledger declarations in `particula/execution/communication.py`.
- [ ] Implement deterministic validation of edge indices, duplicates/conflicts,
  transfer bounds, dimensions, representation support, and physical values.
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

- [ ] Add `particula/gpu/kernels/communication.py` with staged amount-ledger,
  volume-normalization, gas-transfer, and particle-plan/commit kernels.
- [ ] Validate all caller-owned arrays, aliases, devices, dtypes, ranks, values,
  outbound sums, and destination capacity before any state commit.
- [ ] Keep source reads synchronous by staging from pre-node state; do not make
  results depend on edge declaration order.
- [ ] Preserve container/array identities and fixed dimensions across successful
  calls; reuse session scratch rather than allocating in repeated steps.
- [ ] Implement explicit closed-map conservation diagnostics and open-boundary
  source/sink accounting without host readback in the normal path.
- [ ] Preserve per-particle species mass and charge during slot transport and use
  deterministic matching/free-slot selection without resize or compaction.
- [ ] Ensure empty maps, disabled edges, and unchanged volumes are write-free
  after complete validation.

## Tooling / Tests

- [ ] Add `particula/execution/tests/communication_test.py` for declaration and
  validation contracts.
- [ ] Add `particula/gpu/kernels/tests/communication_test.py` for volume, gas,
  particle, capacity, atomicity, identity, and no-op behavior.
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
