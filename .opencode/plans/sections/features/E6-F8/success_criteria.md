# Success Criteria

- [ ] Plan metadata and implementation preserve mandatory E6-F5, E6-F6, and
  E6-F7 dependencies; E6-F9 can consume the intended low-level entry point.
- [ ] Direct Warp activation and kinetic rates, potential events, admission,
  gas removal, and source diagnostics match the independent CPU oracle at
  recorded float64 tolerances.
- [ ] Every successful unscaled case conserves represented particle-plus-gas
  mass per box/species. Scaled cases match `s * pre_total` and conserve intensive
  concentration plus source transfer balance; gas never becomes negative.
- [ ] Slot activation and exhaustion consume E6-F5/E6-F6 contracts, preserve
  fixed shapes/identities, and never resize, compact, or truncate demand.
- [ ] Invalid or unsatisfiable calls fail before any particle, gas, volume,
  request, diagnostic, scratch/work, or RNG write.
- [ ] The implementation performs no hidden CPU/Warp transfer, `.numpy()`
  physics evaluation, CPU fallback, or implicit high-level backend selection.
- [ ] Warp CPU parity tests pass; CUDA tests pass when available and otherwise
  skip cleanly. Changed-code coverage remains at least 80%.
- [ ] Documentation states the bounded physics, ownership, transfer, parity,
  conservation, no-fallback, and deferred-feature contracts accurately.

## Metrics

| Metric | Baseline | Target | Source |
|--------|----------|--------|--------|
| Direct GPU nucleation entry points | 0 | 1 bounded step | Import/API tests |
| Per-box/species conservation error | No GPU evidence | `rtol=1e-12`, `atol=1e-30` target | Parity suite |
| Negative gas after successful finalization | Not applicable | 0 cases | Inventory tests |
| Silent truncated represented demand | Not applicable | 0 cases | Exhaustion tests |
| Hidden transfer or CPU fallback paths | Not applicable | 0 | Source/API regression checks |
| Required Warp CPU case pass rate | 0% | 100% | Focused pytest suite |
