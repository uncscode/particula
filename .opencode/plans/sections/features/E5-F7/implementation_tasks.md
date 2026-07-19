# Implementation Tasks

### Validation Infrastructure — P1 completed in issue #1362

- [x] Freeze literal executable rows (four singleton, six two-way, and one
  four-way) and deferred three-way rows in the test support table.
- [x] Add explicit fp64 one-/two-box, mixed-scale, charged, inactive-gap, and
  active-count fixtures in `_coagulation_validation_support.py`.
- [x] Implement independent NumPy/public-CPU pair and property oracles for
  Brownian, charged, SP2016, ST1956, additive sums, and selector majorants.
- [x] Add host configuration/boundary tests plus lazy Warp-CPU property, pair,
  symmetry, majorant, and exact-zero observations in
  `coagulation_validation_test.py`.
- [ ] Add a reusable per-box/per-species mass balance assertion with a physical
  absolute floor and a separate total-charge balance assertion.
- [ ] Reuse shared `warp_devices()` parameterization so CPU always participates
  and CUDA participates only when available.

### Deterministic and Ownership Matrix

- [x] Table-drive pair symmetry, finite/non-negative values, deterministic
  parity, component sums, and `pair_rate <= majorant` for explicit active pairs.
- [ ] Run all executable rows across zero/one/two/many active particles,
  inactive gaps, mixed radii/masses/charges, and zero-rate conditions.
- [ ] Assert sorted, in-range, disjoint accepted pairs, collision capacity, donor
  clearing, inactive-state preservation, and no cross-box contamination.
- [ ] Assert caller-provided pair/count buffer identity and persistent RNG
  advance/reset semantics for each representative row.
- [ ] Snapshot particle fields, output buffers, and RNG state around invalid
  cross-row inputs to prove validation is atomic before mutation/advancement.

### Stochastic and Device Evidence

- [ ] Define sample counts and independent expected collision aggregates before
  observing implementation results; document the confidence or sigma formula.
- [ ] Execute repeated fresh seeded trials per mechanism row and assert aggregate
  bounds without requiring exact CPU/Warp accepted-pair replay.
- [ ] Apply deterministic legal-pair, capacity, conservation, inactive-slot, and
  ownership assertions to every stochastic trial.
- [ ] Mark tests with `gpu_parity`, `stochastic`, and `cuda` as applicable and
  verify Warp CPU failures cannot be hidden by CUDA availability.
- [ ] Run focused deterministic/stochastic commands and the full coagulation
  suite without slow/performance markers.

### Documentation

- [ ] Publish a matrix mapping each supported row to deterministic, stochastic,
  conservation, edge, and device evidence.
- [ ] Document tolerances, seed/sample policy, Warp CPU requirement, optional
  CUDA behavior, known unsupported rows, and focused reproduction commands.
- [ ] Validate documentation links, command names, pytest markers, and support
  wording for E5-F9 consumption.
