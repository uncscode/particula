# Documentation Updates

- Shipped code-level documentation: `thermodynamics.py` documents the
  concrete-module-only `refresh_vapor_pressure_gpu` import, its validated Warp
  `float64` boundary, constant Pa semantics, canonical Buck water/ice behavior,
  and Buck's reserved/unused parameter slots.
- Migrated the executable GPU direct-kernels quick-start call without adding
  user-facing thermodynamics guidance.
- Updated `AGENTS.md` with the concrete-module GPU vapor-pressure refresh API,
  validated Warp `float64` temperature input, and constant/Buck behavior.
- No feature, roadmap, migration-guide, or other user-facing thermodynamics
  documentation was updated; that work remains deferred to P5.
