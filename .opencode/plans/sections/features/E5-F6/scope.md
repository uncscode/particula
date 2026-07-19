# Scope

E5-F6 extends the mechanism contract established by E5-F1 and the executable
terms delivered by E5-F3, E5-F4, and E5-F5. It registers approved additive
combinations, sums component pair rates, scans compact active pairs for the
exact summed-rate production majorant, and performs one bounded
candidate/acceptance pass for every approved mask.

Status: P1--P4 are implemented. Executable masks are `1`, `2`, `4`, `8`, `3`,
`5`, `6`, `9`, `10`, `12`, and `15`; three-way masks `7`, `11`, `13`, and `14`
remain deferred.

## In Scope

- Define and validate the canonical additive capability matrix after all four
  terms are executable: Brownian, charged, SP2016 sedimentation, and ST1956
  turbulent shear.
- Preserve existing single-term modes and Brownian-plus-charged behavior.
- Support and retain regression evidence for every approved singleton and
  two-way row plus the full Brownian+charged+sedimentation+turbulent-shear row;
  explicitly encode intentionally unsupported three-way rows rather than
  accepting them by accident.
- Compute `total_pair_rate = sum(enabled_pair_rates)` for each sampled pair.
- Compute the production `total_majorant` by scanning compact active pairs for
  the exact maximum summed rate. Retain `sum(enabled_term_majorants)` only as a
  conservative proof/diagnostic bound for every tested active-pair sum.
- Schedule trials and make exactly one acceptance decision per proposal using
  `total_pair_rate / total_majorant`.
- Reuse one active set, collision-pair/count buffers, per-box RNG state, and
  charge-aware apply pass; preserve mass and charge conservation.
- Validate all mechanism-specific required inputs before allocations, launch,
  output mutation, particle mutation, or RNG advancement.
- Add Warp CPU deterministic and bounded stochastic tests; CUDA is optional and
  skips cleanly when unavailable.

## Out of Scope

- New pair physics, majorant derivations, or correction models owned by E5-F2
  through E5-F5.
- DNS turbulence, non-unit sedimentation efficiency, unapproved charged
  variants, binned/continuous-PDF execution, or general CPU-strategy parity.
- Multiple stochastic passes, per-mechanism pair buffers, exact CPU/Warp pair
  replay, or a new public combined-step entry point.
- High-level `Aerosol`/`Runnable` integration, CPU fallback, dynamic slots,
  hidden transfers/synchronization, graph-capture guarantees, adaptive
  stepping, performance redesign, or broad accuracy claims.
- E5-F7's full cross-mechanism release matrix and E5-F9's user-facing example
  and roadmap closeout.
