# Open Questions

- [x] Confirm ideal activity support is limited to molar fraction for E4-F2.
  - Resolved 2026-07-13: yes. Support ideal molar-fraction activity and kappa;
    ideal mass/volume fractions and BAT remain CPU-only.
- [x] Select the composition-dependent surface contract.
  - Resolved 2026-07-13: support static per-species tension and global
    single-phase volume-weighted tension. Defer phase-aware weighting and
    fixed-shape `phase_index` metadata.
- [x] Define numeric mode values and omitted-configuration behavior.
  - Resolved 2026-07-13: P1 defines stable named integer constants for unit
    activity, kappa activity, static surface tension, and global
    volume-weighted tension. Unsupported values fail before mutation, and
    omitted configuration preserves legacy unit-activity, per-species static
    behavior; a supplied malformed sidecar fails before launch or mutation.
- [x] Record scientific `rtol` and `atol` per formula and coupled fixture.
  - Resolved 2026-07-13: begin with `rtol=1e-10` and scale-derived `atol` for
    fp64 Warp CPU and CUDA formula/coupled parity. E4-F6 records any measured,
    case-specific relaxation; invariants use their separate tighter bounds.
- [x] Decide weighted-tension scratch ownership and recomputation.
  - Resolved 2026-07-13: recompute from current device masses every step into a
    step-owned fp64 `(n_particles,)` buffer. This gives one full weighted
    reduction per active particle without exposing reusable scratch state.

No question may be resolved by silently choosing stale/zero E4-F1 pressure,
adding strings/strategy objects to Warp data, or introducing host recomputation.
