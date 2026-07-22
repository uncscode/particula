# Scope

E6-F6 adds CPU and direct Warp exhaustion planning and commit primitives for
fixed-slot particle-resolved boxes, preserving E6-F5 classification and
activation contracts.

## In Scope

- A policy/configuration contract with `resampling=True` and
  `representative_volume_scaling=False` defaults.
- Independent controls and explicit resampling-first precedence when both are
  enabled.
- Deterministic CPU resampling and allocation-stable Warp parity.
- Optional per-box representative-volume scaling with same-direction raw-weight
  and source-demand updates plus explicit pre-scale/represented diagnostics.
- Transactional preflight/plan/commit behavior; both policies disabled,
  malformed state, missing scratch, or unsatisfiable demand fails before writes.
- Sparse, exact-capacity, full, and over-capacity multi-box tests for number,
  species mass, charge, fixed identities, and distribution-preservation targets.
- Caller-owned fixed-shape diagnostics/work buffers and Warp CPU evidence with
  optional CUDA execution.

## Out of Scope

- Dynamic allocation, array resizing/appending, compaction, hidden transfers,
  CPU fallback, or container-schema changes.
- Nucleation rate physics, gas depletion, or source-record construction, which
  belong to E6-F7/E6-F8.
- High-level GPU `Runnable` APIs, backend selection, scheduling, resident loops,
  graph-capture claims, differentiability, and performance claims (Epic G+).
- Exact CPU/CUDA RNG-stream matching; the bounded design is deterministic and
  does not add policy-owned stochastic state.
