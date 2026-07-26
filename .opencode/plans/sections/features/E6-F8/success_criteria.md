# Success Criteria

## P1 completion (#1438)

- [x] Concrete-only frozen configuration and sidecar records retain
  caller-owned arrays without copies or rebinding.
- [x] Private P1 preflight validates schema, device, alias, physical, and
  species/count constraints before any caller write; gates and rejections leave
  supplied state untouched.
- [x] P1 remains unexported and performs no step execution, rate computation,
  transfer, fallback allocation, or mutation.

## P3 completion (#1440)

- [x] Private P3 accepts only finite, nonnegative, exactly integral
  demand-volume products in the inclusive int32 range and uses bounded private
  conversion workspace plus one scalar status readback.
- [x] P3 reuses E6-F5 diagnostics, retains full provisional counts beyond free
  capacity, and writes only caller-owned int32 P3/E6-F5 sidecars by identity.
- [x] Conversion and E6-F5 preflight failures preserve P3/E6-F5 outputs; no
  rollback is claimed after launched asynchronous diagnostic or commit writers.
- [x] P3 has no activation, E6-F6 policy, particle/gas mutation, export, host
  fallback, resize, or hidden transfer.

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
