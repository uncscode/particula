# Success Criteria

- [ ] Independent fp64 fixtures show effective density, Stokes/slip settling
  velocity, and SP2016 pair-rate agreement on Warp CPU within declared
  tolerances.
- [ ] The device pair equation is
  `pi * (r_i + r_j)^2 * abs(v_i - v_j)` and exposes no way to request collision
  efficiency other than 1.
- [ ] Every finite active-pair sedimentation rate is non-negative and no greater
  than the computed majorant in deterministic exhaustive fixtures.
- [ ] Sedimentation-only execution uses E5-F1's bounded single candidate/
  acceptance pass and does not add an independent RNG stream or selector.
- [ ] One-box and multi-box/multi-species tests cover differing environments,
  inactive slots, zero/one/two active particles, and equal settling velocity.
- [ ] Accepted merges preserve total species mass per box, clear donor mass and
  concentration, and retain legal sorted disjoint pair indices.
- [ ] Caller-owned collision buffers and persistent RNG buffers are reused by
  identity and keep documented reset/advancement semantics.
- [ ] Unsupported distributions, non-unit/dynamic efficiency, unavailable
  additive combinations, invalid domains, and shape/dtype/device mismatches
  raise before particle, output-buffer, or RNG mutation.
- [ ] Warp CPU evidence passes when Warp is installed; CUDA runs when available
  and skips cleanly otherwise.
- [ ] Documentation states all support limits without claiming high-level,
  additive, drag-corrected, DNS, fallback, or exact stochastic parity support.
- [ ] Existing Brownian behavior and the public return tuple remain regression-
  clean, and coverage thresholds are not lowered.

## Metrics

| Metric | Baseline | Target | Source |
|--------|----------|--------|--------|
| Executable GPU sedimentation modes | 0 | 1 sedimentation-only SP2016 mode | Capability-matrix tests |
| Collision-efficiency values supported | 0 | Exactly `{1}` | API and rejection tests |
| Pair rates exceeding majorant | Not measured | 0 across exhaustive fixtures | Majorant tests |
| Species-mass conservation error | No sedimentation path | Within declared fp64 tolerance per box/species | End-to-end tests |
| Unsupported-call state mutations | Not measured | 0 | Pre/post snapshot tests |
| Required device evidence | Brownian Warp CPU | Sedimentation Warp CPU; optional CUDA | Pytest device matrix |
