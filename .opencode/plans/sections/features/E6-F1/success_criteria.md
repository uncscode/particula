# Success Criteria

- [ ] The CPU dilution coefficient and concentration-update equations, units,
  accepted shapes, and finite-step semantics are documented and tested.
- [ ] Existing `get_volume_dilution_coefficient()` and
  `get_dilution_rate()` behavior and import paths remain compatible.
- [ ] A public CPU dilution strategy updates particle number concentration and
  scalar or multi-species gas mass concentration against an independent NumPy
  reference at recorded float64 tolerances.
- [ ] A public `Dilution` runnable implements `rate()` and `execute()`, supports
  validated substeps, returns the same aerosol, and composes through `|`.
- [ ] Zero input flow/coefficient and zero elapsed time are exact no-ops for
  particle, gas, and atmospheric state.
- [ ] Particle mass, charge, density, distribution coordinates and
  representation volume remain unchanged for every successful dilution step.
- [ ] Gas names, molar masses, vapor-pressure strategies, partitioning flags,
  and atmospheric temperature/pressure remain unchanged.
- [ ] Successful updates produce finite nonnegative concentrations and preserve
  all input array/container shapes.
- [ ] Invalid volume, flow, coefficient, time, substeps, shapes, types, or
  concentration state fail before any particle or gas mutation.
- [ ] Public symbols are exported through `particula.dynamics` and normal
  top-level Particula usage is covered by an import smoke test.
- [ ] Fast co-located tests pass, changed code meets the configured minimum 80%
  coverage, and coverage thresholds are not lowered.
- [ ] The CPU example executes and documentation records support/deferred
  boundaries without claiming E6-F2 or Epic G functionality.

## Metrics

| Metric | Baseline | Target | Source |
|---|---|---|---|
| Process-level CPU dilution APIs | 0 | Strategy + runnable | Public import tests |
| Particle/gas reference cases passing | Free-function-only | 100% of acceptance matrix | Focused pytest results |
| Exact zero-flow/time no-op failures | Unspecified | 0 | Snapshot tests |
| Protected-field mutations | Unspecified | 0 | Invariant tests |
| Invalid-call partial mutations | Unspecified | 0 | Preflight immutability tests |
| Changed-code test coverage | N/A | >= 80%, no threshold reduction | pytest-cov/CI |
| Downstream parity readiness | No process oracle | Documented deterministic CPU oracle | E6-F2 handoff review |
