# Success Criteria

- [ ] Activation and kinetic strategies reproduce `J=A*C` and `J=K*C^2`
  references after documented SI conversion.
- [ ] Every strategy documents units, citations, injection convention, and a
  closed validity domain; unsupported extrapolation raises before mutation.
- [ ] Source finalization jointly limits events by every participating gas
  species and never produces negative gas concentration.
- [ ] E6-F5 slot and E6-F6 resampling-first/scaling-fallback semantics are
  consumed unchanged; exhausted demand is never silently truncated.
- [ ] Without scaling, successful calls conserve represented particle-plus-gas
  mass per box/species. With scale `s`, final represented totals match
  `s * pre_total`; intensive particle-plus-gas concentration and source transfer
  balance remain conserved at recorded `float64` tolerances.
- [ ] Invalid multi-box calls preserve gas, particles, diagnostics, requests,
  and work state; zero time/rate/precursor/survival are exact no-ops.
- [ ] Builders, factory, imports, and `Nucleation` runnable have fast tests and
  current documentation.
- [ ] E6-F8 has an independent deterministic CPU oracle and frozen source and
  diagnostics contracts.
- [ ] Fast tests, Ruff, and mypy pass without lowering coverage thresholds.

## Metrics

| Metric | Baseline | Target | Source |
|---|---:|---:|---|
| Supported bounded CPU rate laws | 0 | 2 | Strategy tests |
| Negative gas outcomes | Not prevented | 0 | Integration matrix |
| Silent residual demand | Undefined | 0 | Full-slot tests |
| Per-box/species conservation relative error | No implementation | `<=1e-12` for standard fixtures | Independent oracle |
| Changed arrays on rejected calls | No implementation | 0 | Snapshot tests |
| New/changed code coverage | N/A | `>=80%`, threshold unchanged | pytest-cov |
