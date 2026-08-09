# Scope

Issue #1520 completes only E7-F8-P1: a direct-import-only registration and
explicit initialization boundary for process-scoped RNG state.

## In Scope

- Direct-only `StreamKey`, `StreamDescriptor`, and `StreamRegistry` metadata for
  `coagulation` and `wall_loss` in `particula.execution.rng`.
- Exact root-seed/key FNV derivation, canonical two-array manifests, logical-ID
  registration, lane lookup, collision rejection, and immutable lookup results.
- Lazy NumPy/Warp imports, full pre-write array validation, and explicit copying
  of lane-indexed `uint32` words into caller-owned buffers.
- Focused RNG, coverage, and execution export-denial regression coverage.

## Out of Scope

- Exact CPU/Warp/CUDA stochastic trajectory equality or cross-device restart
  bitwise equivalence.
- Resident resource ownership, kernel adapters, scheduling, enable masks,
  checkpoint/restart, reset APIs, and stream advancement policy.
- Rewriting coagulation or wall-loss physics, random algorithms, or selection
  order beyond the minimum interface needed for per-box enablement.
- RNG for deterministic processes, unsupported mechanisms, CPU runnable redesign,
  transport randomness, multi-GPU/distributed streams, or cryptographic RNG.
- Hidden host readback, per-step synchronization, automatic reseeding, silent
  fallback, dynamic particle capacity, graph capture/performance work (Epic H),
  or autodiff work (Epic I).
- A durable file format, arbitrary-object deserialization, remote checkpoint
  storage, or long-term compatibility beyond the declared checkpoint schema.
