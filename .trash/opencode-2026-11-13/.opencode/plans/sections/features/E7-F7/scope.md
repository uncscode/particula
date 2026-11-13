# Scope

E7-F7 supplies the opt-in, prescribed communication layer for E7's resident
single-device multi-box loop. P5 integrates the shipped direct primitives using
one pinned, closed-map resident resource family and E7-F5's deterministic
twelve-node scheduler barrier.

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
- Fixed-capacity particle transport, shipped as concrete-only
  `particula.gpu.kernels.communication.particle_communication_step_gpu` (#1510).
  It preserves per-particle mass/species and signed charge, transfers
  concentration from immutable pre-step state, uses exact population matching or
  ascending pre-step free-slot reservations, and gates its one-kernel commit on
  a complete representable closed-map plan.
- Synchronous, registration-order-independent P3 gas updates from pre-step state
  using caller-owned work buffers; reusable resident scratch is deferred.
- P5 resident integration: pin exactly one GAS or PARTICLES closed-map
  configuration, its map arrays, native work record, and optional final-volume
  sidecar by identity. Acquisition is the sole P1 payload-validation point;
  normal execution performs metadata/identity validation only.
- P5 schedule/lifecycle integration: communication then optional volume
  evolution precede all pre-existing loop nodes in the closed twelve-node graph.
  Both barriers use pre-update volumes and invalidate only `SATURATION_RATIO`.
- P5 checkpoints: controller-created schema-v2 checkpoints preserve no family
  or one complete pinned family and restart into fresh identities; schema-v1
  noncommunication checkpoints remain restart-compatible.
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
