# Scope

Deliver a low-level `dilution_step_gpu` over fixed-shape `WarpParticleData` and
`WarpGasData`, using the E6-F1 CPU finite-step contract as the parity oracle.
P1 (#1395) ships the concrete-module input contract only: scalar/per-box
coefficient metadata validation, no launches/writes, and same-object return.
P2-P4 remain responsible for mutation, full preflight, and parity.

## In Scope

- Scalar and same-device `wp.float64` per-box coefficient inputs; preserve the
  T1 volume/flow-derived units and semantics without requiring host conversion.
- P1's finite, nonnegative scalar coefficient/time validation; valid per-box
  array identity retention and P1 no-write identity return.
- Fixed-shape particle-number and gas-mass concentration kernels for one or
  multiple boxes/species.
- Validated finite, nonnegative coefficient and time inputs, container shapes,
  dtypes, devices, and concentration state before writes.
- Exact zero-coefficient and zero-time no-ops.
- Deterministic Warp CPU parity against an independent implementation of the T1
  oracle, plus optional CUDA evidence.
- Identity/value invariants for caller inputs and all nondiluted container
  fields; public low-level export and direct-use documentation (P2/P5).

## Out of Scope

- Reimplementing or changing the T1 CPU strategy/runnable contract.
- Hidden CPU/GPU conversion, synchronization, fallback, or container cloning.
- A GPU `Runnable`, backend selector, scheduler, multi-box transport, dynamic
  allocation/resizing, graph capture, autodiff, or performance claims.
- Wall loss, nucleation, slot activation/exhaustion, or integrated Epic F
  sequencing, which belong to sibling E6 features.
