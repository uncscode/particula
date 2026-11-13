# Infrastructure Reuse

- Reproduce ideal and kappa equations from
  `particula/particles/properties/activity_module.py:16-258`, including dry and
  zero-solute guards.
- Match effective surface behavior in
  `particula/particles/surface_strategies.py:27-162,304-525`; preserve the
  static per-species path and define the selected composition weighting.
- Reuse `kelvin_radius_wp`, `kelvin_term_wp`, and
  `partial_pressure_delta_wp` from
  `particula/gpu/properties/particle_properties.py:159-231`.
- Extend validation and launch ordering in
  `particula/gpu/kernels/condensation.py:387-570`; integrate physics in the
  mass-transfer kernel near lines 58-178.
- Consume E4-F1's refreshed `WarpGasData.vapor_pressure` contract described in
  `.opencode/plans/sections/features/E4-F1/architecture_design.md`.
- Keep ownership and shapes from `particula/gpu/warp_types.py:24-169`.
- Extend independent references and fixed-buffer fixtures in
  `particula/gpu/kernels/tests/_condensation_test_support.py:580-883`.
