# Scope

E7-F9 is the closeout gate for E7. It consumes all E7-F1 through E7-F8
contracts, adds only the diagnostics and cross-feature evidence needed by Track
T9, publishes the complete example and support matrix, and preserves the issue
#1451 exit bar without reopening process physics or earlier API decisions.

## In Scope

- Optional GPU-side reductions for total particle-plus-gas species mass,
  particle number concentration, latent-heat energy, and conservation residuals.
- A frozen, versioned checkpoint evidence schema covering physical state,
  semantic metadata, dimensions, process configuration, counters, and RNG state.
- Multi-timestep resident-loop regressions for deterministic process order,
  current thermodynamic state, identity stability, and checkpoint-only transfers.
- Independent-box parity/isolation and larger particle-resolved multi-box cases.
- Prescribed advection, dilution, mixing, expansion, and volume-evolution
  regressions with explicit open/closed accounting and CPU references.
- Same-backend checkpoint/restart and stable logical-box RNG regressions.
- Warp CPU as the required routine matrix; optional CUDA rows that skip cleanly.
- A documented complete example, support/limitations contract, validation
  commands, recorded tolerances, and dated Epic G closeout evidence.

## Out of Scope

- New process physics, GPU staggered condensation, unsupported coagulation modes,
  dynamic resizing/compaction, multi-GPU/distributed execution, or CFD coupling.
- Silent fallback, hidden transfer/synchronization, per-step host diagnostics, or
  exact CPU/Warp/CUDA stochastic trajectory equality.
- Graph capture, profiling, benchmarks, scaling claims, or optimization (Epic H).
- Autodiff, inverse modeling, and optimization workflows (Epic I).
- Redesigning contracts owned by E7-F1 through E7-F8; discovered defects return
  to the owning track unless a narrow closeout regression is sufficient.
