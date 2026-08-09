# Scope

E7-F7 supplies the opt-in, prescribed communication layer for E7's resident
single-device multi-box loop. Its shipped P2 volume and P3 gas primitives are
isolated direct device-resident operations over caller-owned fixed-capacity
particle/gas state; later phases may place proven communication operations in
E7-F5 scheduler nodes under E7-F6 capability and failure policy.

## In Scope

- Immutable fixed-shape or fixed-capacity communication-map/configuration types
  with explicit source, destination, enabled-edge, transfer, mixing, and volume
  inputs.
- Read-only validation of indices, shapes, dtypes, devices, aliases, finite
  physical values, topology, and resident dimensions before mutation. P1
  deliberately defers population-dependent outbound-overdraw and destination
  capacity checks to the writer phases that receive source inventory and time.
- Prescribed per-box positive volume updates owned by `ParticleData.volume`,
  including shipped direct P2 concentration renormalization for
  expansion/compression and retention of protected fields and identities.
- Conservative gas advection and simple mixing using immutable extensive amount
  ledgers: shipped as concrete-only
  `particula.gpu.kernels.communication.gas_communication_step_gpu` (#1509).
  It supports closed maps plus declared `-1` source/sink endpoints with
  caller-owned accounting ledgers, rejects aggregate overdraw, and commits gas
  concentration once without changing volume or particle state.
- Fixed-capacity particle transport that preserves per-particle mass/species and
  charge state, transfers concentration/count, and fails before commit when the
  prescribed destination cannot represent transported slots.
- Synchronous, registration-order-independent P3 gas updates from pre-step state
  using caller-owned work buffers; reusable resident scratch is deferred.
- E7-F4 resource registration and checkpoint inclusion, E7-F5 deterministic
  communication-node placement, and E7-F6 capability/error integration remain
  deferred; P3 accepts caller-owned work storage rather than resident resources.
- Independent-box no-op behavior, one-dimensional neighbor maps, user-defined
  box pairs, Warp CPU parity, optional CUDA evidence, and explicit conservation
  tests.
- Co-located unit/contract tests and final user/developer documentation.

## Out of Scope

- Full CFD coupling, pressure/velocity solvers, adaptive meshes, distributed or
  multi-GPU communication, or network transport.
- Dynamic particle storage resizing, compaction, implicit slot creation, or
  silent clipping when destination capacity is insufficient.
- Hidden CPU/GPU transfer, implicit synchronization, runtime retry, or silent
  CPU fallback.
- Rewriting condensation, coagulation, dilution, wall-loss, nucleation, or
  thermodynamic kernel physics.
- Graph capture, performance claims, profiling, and optimization (Epic H).
- Autodiff or inverse workflows (Epic I).
- Per-box stochastic stream policy, which belongs to E7-F8, and full-loop
  publication/regression ownership, which belongs to E7-F9.

The feature must preserve issue #1451 validation rules: fixed shapes and
identities, explicit ownership and synchronization, CPU references, Warp CPU as
the routine backend, clean optional-CUDA skips, and no unsupported physics claim.
