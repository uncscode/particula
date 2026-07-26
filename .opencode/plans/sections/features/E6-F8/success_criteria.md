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

## P4 completion (#1441)

- [x] P4 preserves P2 accepted demand and P3 full counts/diagnostics as immutable
  handoffs, selects fully viable resampling before scaling fallback, and never
  truncates unmet demand.
- [x] P4 derives final counts from post-policy demand-volume products and writes
  caller-owned final demand/count/ascending-free-prefix diagnostics by identity.
- [x] Expected all-box failures reject before P4 writes or E6-F6 primitive entry
  and preserve particle/gas data plus every P2/P3/P4/nested sidecar.
- [x] The distinct entered-primitive planning/commit failure boundary is
  documented and tested without claiming cross-primitive rollback.
- [x] P4 remains concrete-only and adds no activation, particle/gas mutation,
   public export, E6-F9 integration, fallback, resize, or hidden transfer.

## P5 completion (#1442)

- [x] `nucleation_step_gpu` is lazily exported as the sole kernel-package
  nucleation symbol; concrete configuration, records, sidecars, and helpers
  remain unexported.
- [x] P1--P4 complete before bounded P5 handoff validation and one fused commit
  initializes only finalized selected slots and removes matching finalized gas.
- [x] Successful calls return identical particle/gas containers; zero-demand
  rows are write-free and fixed schemas, protected fields, and preexisting slots
  are preserved.
- [x] Malformed/rebound P5 handoffs and other precommit failures preserve
  particle/gas state, while P2--P4 and entered-E6-F6 sidecars retain their
  documented mutation limits.
- [x] The direct boundary adds no hidden transfer, CPU fallback, resize,
  compaction, Runnable, or E6-F9 integration.

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
