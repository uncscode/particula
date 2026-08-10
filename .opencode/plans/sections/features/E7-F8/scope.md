# Scope

Issues #1520--#1526 completed E7-F8-P1--P7: concrete-only resident RNG
lifecycle support, same-device invariance coverage, and P7 documentation for
Brownian coagulation and wall loss built on P1 registration and initialization.

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
  and literal `initialize_rng=False`.
- One independently initialized, session-owned wall-loss `(n_boxes,)`
  `wp.uint32` sidecar with canonical-manifest initialization, exact resource
  identity, and coagulation nonaliasing.
- Scheduler-resolved selected-box wall-loss dispatch through the unchanged
  direct-kernel interface. Disabled, prelaunch-skipped, zero-time, and valid
  no-work lanes preserve particle/RNG state; post-launch failure retains the
  existing scheduler fault lifecycle without rollback.
- Direct-only frozen `StreamManifest` and published-stream inspection metadata;
  inspection exposes no arrays, pointers, current words, or device state.
- Explicit selected-lane initialization in `rng.py`, published-sidecar-only
  initialization in `gpu_resources.py`, and ACTIVE exact
  session/registry/closed-guard composition through `ResidentSession`.
- Exact-tuple process/logical-box selector validation before writers, including
  write-free empty selections and explicit unpublished-process rejection.
- Same-device resident adapter regressions proving `box-a` stream/output
  invariance across active, removed, no-work, and physical-lane-permuted
   unrelated-box arrangements, plus wall-loss selected/no-op/rejection gating.
- Schema-v3 optional checkpoint continuation for the canonical published
  `coagulation` and `wall_loss` streams: immutable metadata and little-endian
  `(n_boxes,)` `uint32` current-word payloads are the sole RNG-byte authority.
- One-synchronization bounded RNG capture, fail-closed v3 preflight, and fresh
  exact-device restoration without stream acquisition or reseeding. Schema-v1
  and schema-v2 (including valid communication records) retain their existing
  compatible restart behavior.
- Checkpoint/restart preflight, split-run continuation, reset-after-restart,
  malformed-record rejection, capture/readback failure, and fresh-construction
  recovery coverage in the execution test suite.

## Out of Scope

- Exact CPU/Warp/CUDA stochastic trajectory equality or cross-device restart
  bitwise equivalence.
- Full-box invariance guarantees beyond the covered same-device resident
  Brownian and selected neutral wall-loss regression matrix.
- Durable file serialization, arbitrary-object deserialization, remote
  checkpoint storage, migration, and cross-device RNG replay. Continuation is
  in-memory, schema-v3-only, and exact-device only.
- Rewriting coagulation or wall-loss physics, random algorithms, direct-kernel
  signatures, host readback, transfer, or synchronization behavior.
- RNG for deterministic processes, unsupported mechanisms, CPU runnable redesign,
  transport randomness, multi-GPU/distributed streams, or cryptographic RNG.
- Hidden host readback, per-step synchronization, automatic reseeding, silent
  fallback, dynamic particle capacity, graph capture/performance work (Epic H),
  or autodiff work (Epic I).
