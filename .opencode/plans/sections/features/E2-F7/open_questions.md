# E2-F7 Open Questions

## Resolved Answers

- E2-F2 should finalize `EnvironmentData.temperature` and
  `EnvironmentData.pressure` as `(n_boxes,)`, plus
  `EnvironmentData.saturation_ratio` as `(n_boxes, n_species)`. E2-F3 should
  mirror those fields and shapes in `WarpEnvironmentData`.
- The first production GPU condensation integration should update both particle
  and gas concentration for physical completeness. E2-F7 should add or expand a
  phase for gas-coupled updates and conservation checks; if that exceeds the
  feature size, split the production gas-coupled implementation into a follow-up
  feature and keep E2-F7 as the measured foundation.
- Stable timestep classification should use a combined metric: finite and
  non-negative state, bounded fractional particle-mass change, monotonic movement
  toward equilibrium where applicable, gas-particle mass conservation, and CPU
  parity within documented tolerances.
- Hard non-negative clamps may remain in guarded non-differentiable production
  paths. Differentiable or optimization workflows require a documented smooth or
  guarded alternative before they are advertised as supported.
- Stiffness evidence should remain `fp64` by default until E2-F6 closes the
  precision envelope. Lower-precision runs may be included only as exploratory
  evidence and must be labeled as such.
- Broad stiffness sweeps should live outside default CI as slow pytest coverage
  or a documentation-generation script. Keep at least one fast Warp CPU test that
  ties the recommendation to executable behavior.
