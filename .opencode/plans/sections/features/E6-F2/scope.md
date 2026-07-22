# Scope

Deliver a low-level `dilution_step_gpu` over fixed-shape `WarpParticleData` and
`WarpGasData`, using the E6-F1 CPU finite-step contract as the parity oracle.
P1 (#1395) shipped the concrete-module input contract. P2 (#1396) ships the
in-place concentration mutation and package export; P3-P4 retain full
preflight/rollback and broader parity evidence.

## In Scope

- Scalar and same-device `wp.float64` per-box coefficient inputs; preserve the
  T1 volume/flow-derived units and semantics without requiring host conversion.
- P1's finite, nonnegative scalar coefficient/time validation; valid per-box
  array identity retention and P1 no-write identity return.
- Fixed-shape particle-number and gas-mass concentration kernels for one or
  multiple boxes/species, applying `exp(-alpha * time_step)` in place.
- Scalar coefficient/time validation plus launch-safety container shape, dtype,
  and device metadata validation before non-no-op writes.
- Exact zero-coefficient and zero-time no-ops.
- Identity/value invariants for caller inputs and all nondiluted container
  fields, and the sole public low-level export through `particula.gpu.kernels`.

## Out of Scope

- Reimplementing or changing the T1 CPU strategy/runnable contract.
- Hidden CPU/GPU conversion, synchronization, fallback, or container cloning.
- A GPU `Runnable`, backend selector, scheduler, multi-box transport, dynamic
  allocation/resizing, graph capture, autodiff, or performance claims.
- Wall loss, nucleation, slot activation/exhaustion, or integrated Epic F
  sequencing, which belong to sibling E6 features.
- Per-box coefficient-value validation, complete concentration/container
  preflight, atomic rollback, and broader CPU/Warp parity, which remain P3/P4.
