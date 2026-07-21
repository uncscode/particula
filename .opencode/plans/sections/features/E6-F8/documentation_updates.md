# Documentation Updates

- Update `docs/Theory/Technical/Dynamics/Nucleation_Equations.md` with the exact
  CPU-to-Warp correspondence, SI units, validity bounds, admission equation,
  represented-mass accounting, and unsupported physics.
- Add or update a `docs/Features/` direct GPU nucleation page documenting
  `nucleation_step_gpu`, concrete-module configuration/scratch APIs, fixed
  shapes, dtypes, devices, ownership, mutation, and failure ordering.
- Add an explicit CPU-to-Warp setup and restore example under
  `docs/Examples/Nucleation/`; keep transfers outside the direct step and make
  Warp CPU the default documented backend.
- Update `AGENTS.md` with intended imports, sidecar contracts, E6-F5/F6/F7
  dependencies, conservation/parity commands, and no-fallback boundaries.
- Cross-link E6-F7's CPU reference and E6-F9's integrated process example;
  state that E6-F8 does not provide a high-level GPU runnable or scheduler.
- Update E6 and E6-F8 plan sections with final issue numbers, measured
  tolerances, shipped status, and any resolved sidecar naming decisions.
