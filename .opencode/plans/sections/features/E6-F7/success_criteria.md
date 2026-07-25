# Success Criteria

- [x] Activation and kinetic strategies reproduce `J=A*C` and `J=K*C^2`
  references after documented SI conversion.
- [x] Every strategy documents units, citations, injection convention, and a
  closed validity domain; unsupported extrapolation raises before mutation.
- [x] Source finalization jointly limits events by every participating gas
  species, produces finite nonnegative provisional demand within inventory,
  and does not mutate gas concentration.
- [x] E6-F5 slot and E6-F6 resampling-first/scaling-fallback semantics are
  consumed unchanged on detached P3 staging; exhausted demand is not silently
  truncated.
- [x] P3 validates per-box/species particle-plus-gas conservation before commit;
  scaled rows use `particle_post + gas_post = s * pre_total`.
- [x] P3 invalid calls preserve caller particle and gas arrays; capacity, no-op,
  scaling, and atomic-rejection tests cover the transaction boundary.
- [x] Strict builders, factory, immutable source-selection metadata, and approved
   P4 imports have regression coverage; P2/P3 exports remain absent.
- [x] CPU-only single-box `Nucleation` and immutable `NucleationCommitConfig`
  have fast topology, validation, identity, sequencing, and failure-boundary
  tests; P2/P3 remain concrete-only. User-facing documentation remains P7 scope.
- [x] P6 has an independent deterministic NumPy `float64` P2/P3 oracle and
  self-contained integration matrix covering multi-box/multi-species source and
  diagnostics contracts without production expected-value helpers.
- [x] P6 snapshot coverage proves P2/P3 preflight and no-viable-policy failures
  preserve accessible particle, gas, input, record, and configuration state;
  P5 regressions retain single-box gas-coupling and backing-identity coverage.
- [x] CPU feature/theory/example documentation defines public and concrete-only
  boundaries, supported CPU scope, conservation, and deferred GPU work.
- [ ] Fast tests, Ruff, and mypy pass without lowering coverage thresholds.

## Metrics

| Metric | Baseline | Target | Source |
|---|---:|---:|---|
| Supported bounded CPU rate laws | 0 | 2 shipped | Strategy tests |
| Negative gas outcomes | Not prevented | 0 | Integration matrix |
| Silent residual demand | Undefined | 0 | Full-slot tests |
| Per-box/species conservation relative error | No implementation | `<=1e-12` for standard fixtures | Independent oracle |
| Changed arrays on rejected calls | No implementation | 0 | Snapshot tests |
| New/changed code coverage | N/A | `>=80%`, threshold unchanged | pytest-cov |
