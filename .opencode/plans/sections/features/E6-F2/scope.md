# Scope

Deliver a low-level `dilution_step_gpu` over fixed-shape `WarpParticleData` and
`WarpGasData`, using the E6-F1 CPU finite-step contract as the parity oracle.
P1 (#1395) shipped the concrete-module input contract. P2 (#1396) shipped the
in-place concentration mutation and package export. P3 (#1397) shipped full
ordered read-only preflight; launched-kernel rollback and broader parity
evidence remain P4 work.

## In Scope

- Scalar and same-device `wp.float64` per-box coefficient inputs; preserve the
  T1 volume/flow-derived units and semantics without requiring host conversion.
- P1's finite, nonnegative scalar coefficient/time validation; valid per-box
  array identity retention and P1 no-write identity return.
- Fixed-shape particle-number and gas-mass concentration kernels for one or
  multiple boxes/species, applying `exp(-alpha * time_step)` in place.
- Complete ordered preflight before all no-op returns, allocation, and launches:
  exact same-device float64 Warp schemas plus finite nonnegative per-box
  coefficient and concentration values.
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
- Rollback after a successfully launched kernel failure and broader CPU/Warp
  parity evidence, which remain P4.
