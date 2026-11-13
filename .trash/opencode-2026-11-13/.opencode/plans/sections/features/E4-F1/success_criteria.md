# Success Criteria

- [x] Constant and Buck have stable numeric mode definitions and fixed parameter
  reservations; constant uses parameter zero in Pa and Buck slots are
  reserved/unused by the canonical evaluator.
- [x] Configuration is numeric, fixed-shape, species-indexed, and validated for
  modes, parameters, shape, dtype, ordering contract, species count, and device.
- [x] Warp constant and Buck formulas match CPU references below, at, and above
  freezing within explicit test tolerances.
- [x] Refresh writes a complete `(n_boxes, n_species)` `float64` pressure buffer
  on-device from current `(n_boxes,)` temperature.
- [x] `condensation_step_gpu()` refreshes pressure exactly once after validation
  and before environment preparation and its mass-transfer step; the primitive
  remains suitable for E4-F3 to call before every future substep.
- [x] Changing GPU temperature between calls updates vapor pressure and the next
  calculation without CPU transfer or host recomputation.
- [x] Missing/invalid configuration fails early under the selected compatibility
  contract and does not mutate pressure, gas concentration, or particle mass.
- [x] Existing positional arguments remain source-compatible; the required new
  sidecar is keyword-only and omission raises `ValueError`.
- [x] Co-located Warp CPU parity/API/no-mutation tests cover the refresh boundary;
  CUDA parity remains optional and skips cleanly when unavailable.
- [x] Scalar, direct Warp-array, and explicit environment inputs select the
   current normalized per-box temperature; direct `wp.float32` is supported by a
   device-local float64 cast.
- [x] Public Warp integration tests reuse caller-owned thermodynamics,
  vapor-pressure, and mass-transfer buffers across two calls, prove the legacy
  positional mass-transfer slot, and snapshot all caller-owned outputs for
  missing-sidecar and optional CUDA cross-device atomic failures.

## Metrics

| Metric | Baseline | Target | Source |
|---|---|---|---|
| Supported on-device models | 0 | 2 (constant, Buck) | Shipped GPU formula tests |
| Refreshed output coverage | Static/possibly zero | 100% of box-species cells per call | Shipped shape/value tests |
| CPU transfer needed after temperature update | Required for host refresh | 0 | Repeated-call integration test |
| Invalid-input post-error mutations | Not guaranteed | 0 | Snapshot regression tests |
| CPU parity cases | 0 | Constant plus Buck ice/boundary/water and mixed species | Parity tests |
