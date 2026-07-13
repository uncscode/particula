# Open Questions

- [x] Which effective surface-tension mode is the minimum supported E4-F2
  contract?
  - Resolved 2026-07-13: static per-species tension and global single-phase
    volume-weighted tension. Phase-aware weighting and `phase_index` metadata
    are deferred.
- [x] Should numeric thermodynamic configuration live in a dedicated Warp
  struct or validated parallel arrays?
  - Resolved 2026-07-13: expose one keyword-only typed operation sidecar that
    owns validated species-indexed `int32` mode and `float64` parameter arrays.
    Do not add process configuration to `WarpGasData`.
- [x] Does the public call return only whole-call energy, or also an explicitly
  named final-substep diagnostic?
  - Resolved 2026-07-13: expose only signed whole-call energy through optional
    caller-owned output storage. Preserve the default two-item return and do
    not expose a final-substep diagnostic.
- [x] What deterministic allocation rule resolves simultaneous particle demand
  when gas inventory is limiting?
  - Resolved 2026-07-13: follow CPU `apply_condensation_limit()` semantics.
    Account for simultaneous evaporation, then proportionally scale all
    positive requests for each box/species when inventory is insufficient.
- [x] Which quantities, if any, are differentiable across clamps and inventory
  gates in the bounded autodiff claim?
  - Resolved 2026-07-13: no differentiability is claimed across clamps, gate
    transitions, inventory scaling, or final in-place mutation. E4-F6 may claim
    only an out-of-place smooth-interior rate slice with inactive bounds.
- [x] What numerical tolerances are accepted for each physics mode on Warp CPU
  versus CUDA?
  - Resolved 2026-07-13: use an initial fp64 parity target of `rtol=1e-10` with
    a scale-derived `atol` on Warp CPU and CUDA. E4-F6 records any empirically
    justified case/device relaxation. Conservation remains separate at
    `rtol=1e-12` and `atol=max(1e-18, scale * eps)`; the stiffness-study
    `5e-2` bound is not a production qualification tolerance.
- [x] Should the M/L labels in `milestones_timeline.md` denote aggregate
  workstream scope rather than implementation-phase size? (reviewer:
  plan-review-sizing)
  - Resolved 2026-07-13: yes. They estimate aggregate milestone workstreams,
    matching E2/E3 convention; child XS/S labels remain implementation-phase
    estimates and do not require splitting.
