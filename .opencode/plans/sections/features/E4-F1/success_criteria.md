# Success Criteria

- [ ] Constant and Buck are the only initially supported explicit models, with
  stable numeric mode definitions and documented parameters/units.
- [ ] Configuration is numeric, fixed-shape, species-indexed, and validated for
  modes, parameters, shape, dtype, ordering contract, species count, and device.
- [ ] Warp constant and Buck formulas match CPU references below, at, and above
  freezing within explicit test tolerances.
- [ ] Refresh writes a complete `(n_boxes, n_species)` `float64` pressure buffer
  on-device from current `(n_boxes,)` temperature.
- [ ] `condensation_step_gpu()` refreshes pressure before its mass-transfer step;
  the primitive is suitable for E4-F3 to call before every future substep.
- [ ] Changing GPU temperature between calls updates vapor pressure and the next
  calculation without CPU transfer or host recomputation.
- [ ] Missing/invalid configuration fails early under the selected compatibility
  contract and does not mutate pressure, gas concentration, or particle mass.
- [ ] Existing positional API calls remain source-compatible.
- [ ] Warp CPU tests pass; CUDA parity passes when CUDA is available and otherwise
  skips cleanly.

## Metrics

| Metric | Baseline | Target | Source |
|---|---|---|---|
| Supported on-device models | 0 | 2 (constant, Buck) | GPU formula tests |
| Refreshed output coverage | Static/possibly zero | 100% of box-species cells per call | Shape/value tests |
| CPU transfer needed after temperature update | Required for host refresh | 0 | Repeated-call integration test |
| Invalid-input post-error mutations | Not guaranteed | 0 | Snapshot regression tests |
| CPU parity cases | 0 | Constant plus Buck ice/boundary/water and mixed species | Parity tests |
