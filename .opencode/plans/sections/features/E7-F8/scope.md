# Scope

E7-F8 defines process-scoped, persistent random streams for stable logical box
identities and integrates those streams with selected Brownian coagulation,
wall-loss scheduling, resident resources, and checkpoint/restart behavior.

## In Scope

- A typed stream identity containing schema version, stochastic process ID, and
  stable caller-visible logical box ID.
- Deterministic derivation of initial per-process/per-box `uint32` Warp state
  from an explicit root seed and stream identity.
- E7-F3 coagulation and existing direct-Warp wall-loss RNG integration through
  E7-F4 resident resources and E7-F5 canonical scheduling.
- Explicit initialize, reset, and read-only metadata/inspection operations at
  documented setup or checkpoint boundaries.
- No advancement for a process/box execution that is disabled or skipped before
  launch; unrelated box insertion, removal, and storage reordering invariance.
- Versioned checkpoint payloads that preserve stream IDs, current mutable state,
  root-seed metadata, and process ownership; exact same-backend restart tests.
- Warp CPU baseline tests, optional CUDA rows, validation/failure tests, and
  development documentation.

## Out of Scope

- Exact CPU/Warp/CUDA stochastic trajectory equality or cross-device restart
  bitwise equivalence.
- Rewriting coagulation or wall-loss physics, random algorithms, or selection
  order beyond the minimum interface needed for per-box enablement.
- RNG for deterministic processes, unsupported mechanisms, CPU runnable redesign,
  transport randomness, multi-GPU/distributed streams, or cryptographic RNG.
- Hidden host readback, per-step synchronization, automatic reseeding, silent
  fallback, dynamic particle capacity, graph capture/performance work (Epic H),
  or autodiff work (Epic I).
- A durable file format, arbitrary-object deserialization, remote checkpoint
  storage, or long-term compatibility beyond the declared checkpoint schema.
