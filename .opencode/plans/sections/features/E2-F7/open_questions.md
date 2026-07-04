# E2-F7 Open Questions

- What exact `EnvironmentData` field names/shapes will E2-F2 finalize, and how
  will E2-F3 mirror them in `WarpEnvironmentData` for per-box temperature and
  pressure?
- Should the first production GPU integration update gas concentration, or is a
  particle-only integration foundation acceptable as an intermediate step?
- What threshold should define an explicitly stable timestep: fractional mass
  change, monotonic approach to equilibrium, CPU parity tolerance, or a
  combined metric?
- Should hard non-negative clamps remain in the differentiable path, or should
  a guarded/smooth alternative be required before optimization workflows?
- Does E2-F6 permit any lower-precision exploratory results, or should all
  stiffness evidence remain fp64-only until the precision envelope is closed?
- Where should broad stiffness sweeps live if they are too slow for default CI:
  slow pytest markers, a documentation generation script, or a separate
  benchmark harness?
