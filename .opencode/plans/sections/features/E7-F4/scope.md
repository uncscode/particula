# Scope

E7-F4 creates the resident-state and transfer lifecycle used by later E7
schedulers. It packages existing fixed-shape Warp containers, immutable session
metadata, process resources, and explicit restore boundaries without changing
the behavior or signatures of shipped direct process kernels.

## In Scope

- A typed GPU resident-session model under the E7-F1 execution layer.
- Explicit setup from validated CPU `ParticleData`, `GasData`, and
  `EnvironmentData` using one conversion call per container.
- Preservation of ordered gas names and other CPU-only metadata outside Warp
  structs.
- A fixed-shape sidecar registry for condensation scratch and thermodynamics,
  coagulation output/RNG arrays, wall-loss RNG arrays, and nucleation planning,
  diagnostics, and exhaustion resources.
- Device, dtype, shape, ownership, identity, and lifecycle validation before a
  session becomes active.
- Normal-step lifecycle hooks that neither synchronize nor restore bulk state.
- Explicit checkpoint and finalization operations with one synchronization
  followed by `sync=False` particle, gas, and environment restores.
- Restart setup from a checkpoint, including opaque mutable sidecar/RNG payloads
  needed by downstream process adapters.
- Clear preflight, launched-work failure, faulted-session, close, and
  idempotency semantics.
- Warp CPU tests, no-intermediate-transfer spies, identity/shape tests,
  checkpoint/restart tests, and optional CUDA rows.

## Out of Scope

- Process ordering, environment evolution, or derived thermodynamic refresh
  scheduling (E7-F5).
- Condensation or Brownian coagulation adapter physics (E7-F2 and E7-F3).
- Per-box RNG stream identity, split/reset policy, and stochastic invariance
  across box reorderings (E7-F8); E7-F4 only stores and restores opaque state.
- Multi-box transport, communication, mixing, or volume evolution (E7-F7).
- Silent CPU fallback, hidden transfers, automatic device migration, or runtime
  retry on another backend.
- Disk/file serialization formats, remote checkpoints, or incremental/delta
  checkpoints.
- Dynamic resizing, compaction, multi-GPU/distributed execution, graph capture,
  profiling, performance claims, autodiff, or unsupported physics expansion.
