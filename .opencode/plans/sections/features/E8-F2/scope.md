# Scope

E8-F2 delivers the T2 setup/enqueue split for the complete resident timestep.
One-time setup performs host-side validation, normalization, dependency
resolution, and prepared-plan construction. The prepared path enqueues only
the already-resolved fixed device operations. Both boundaries are
concrete-only and retain exact caller-owned resident identities.

## In Scope

- Define immutable prepared-enqueue carriers tied to the exact E8-F1 capture
  lifecycle signature and `ResidentSimulationRequest`.
- Move repeated scheduler metadata checks and executor construction into an
  explicit setup operation that is forbidden during capture.
- Provide validation-free, allocation-free enqueue seams for resident state
  updates, thermodynamic refreshes, diagnostics, communication, volume
  evolution, condensation, coagulation, dilution, wall loss, and nucleation.
- Preserve the canonical twelve-node order, refresh windows, selected-box
  semantics, persistent RNG advancement, empty/no-op behavior, and resident
  writer-failure classification.
- Retain existing public direct-kernel entry points as validate-then-enqueue
  wrappers; prepared resident paths remain private/concrete-only.
- Add CPU/Warp-CPU contract tests for setup and enqueue structure, plus
  CUDA-gated capture smoke evidence that skips cleanly when unavailable.

## Out of Scope

- Owning the graph lifecycle, compatibility signature, invalidation, or
  recapture policy established by E8-F1.
- Completing the reusable registry inventory or changing buffer ownership;
  that is E8-F3 scope.
- Three-way scientific parity, scaling benchmarks, memory modeling, examples,
  profiling, or performance claims owned by E8-F4 through E8-F8.
- Dynamic shapes, process selection, communication-map changes, allocation,
  payload readback, synchronization, RNG reseeding, fallback, retry, automatic
  recapture, device migration, resizing, compaction, new physics, or public API
  expansion inside the enqueue boundary.
