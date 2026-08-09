# Scope

Issue #1520 completed E7-F8-P1, and issue #1521 completed E7-F8-P2: a
concrete-only resident Brownian-coagulation stream lifecycle built on P1
registration and initialization.

## In Scope

- Direct-only `StreamKey`, `StreamDescriptor`, and `StreamRegistry` metadata for
  `coagulation` and `wall_loss` in `particula.execution.rng`.
- Exact root-seed/key FNV derivation, canonical two-array manifests, logical-ID
  registration, lane lookup, collision rejection, and immutable lookup results.
- Lazy NumPy/Warp imports, full pre-write array validation, and explicit copying
  of lane-indexed `uint32` words into caller-owned buffers.
- Focused RNG, coverage, and execution export-denial regression coverage.
- Concrete resident stream metadata on `ResidentSession`; one P1-derived,
  `(n_boxes,)` `wp.uint32` coagulation sidecar initialized on first compatible
  resource acquisition and retained by identity.
- Resident Brownian adapter/scheduler dispatch with the exact published sidecar
  and literal `initialize_rng=False`; fail-closed checkpoint/finalize rejection
  once that sidecar is published.

## Out of Scope

- Exact CPU/Warp/CUDA stochastic trajectory equality or cross-device restart
  bitwise equivalence.
- Wall-loss resident RNG ownership, enable-mask/invariance work, generic
  reset/inspection APIs, and stream advancement policy beyond Brownian calls.
- RNG serialization, checkpoint persistence, and restart continuation. A
  checkpoint or finalize attempt after coagulation-sidecar publication rejects
  before payload conversion or enumeration.
- Rewriting coagulation or wall-loss physics, random algorithms, or selection
  order beyond the minimum interface needed for per-box enablement.
- RNG for deterministic processes, unsupported mechanisms, CPU runnable redesign,
  transport randomness, multi-GPU/distributed streams, or cryptographic RNG.
- Hidden host readback, per-step synchronization, automatic reseeding, silent
  fallback, dynamic particle capacity, graph capture/performance work (Epic H),
  or autodiff work (Epic I).
- A durable file format, arbitrary-object deserialization, remote checkpoint
  storage, or long-term compatibility beyond the declared checkpoint schema.
