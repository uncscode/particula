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
- [x] Add a reusable per-box/per-species mass balance assertion with a physical
  absolute floor and a separate total-charge balance assertion.
- [x] Reuse shared `warp_devices()` enumeration so CPU participates when Warp
  is installed and CUDA participates only when available.

### Deterministic and Ownership Matrix

- [x] Table-drive pair symmetry, finite/non-negative values, deterministic
  parity, component sums, and `pair_rate <= majorant` for explicit active pairs.
- [x] Run all executable rows across one-/two-box and one-/two-species
  materializations, sparse/two-active boundaries, inactive gaps, mixed
  radii/masses/charges, and applicable zero-rate conditions.
- [x] Assert sorted, in-range, disjoint accepted pairs, collision capacity, donor
  clearing, inactive-state preservation, and box-local state integrity.
- [x] Assert caller-provided pair/count buffer identity and persistent RNG
  initialization/advance semantics for representative public rows.
- [x] Snapshot particle fields, output buffers, and RNG state around deferred
  and selected invalid cross-row inputs to prove preflight non-mutation.

### Stochastic and Device Evidence

- [x] Define 100-sample fresh-seed cases and independent initial-state collision
  aggregates with the predeclared `3 * sqrt(expected_mean)` formula.
- [x] Execute repeated fresh seeded public-step trials for every executable mask
  and assert aggregate bounds without requiring exact CPU/Warp pair replay.
- [x] Apply deterministic legal-pair, capacity, conservation, inactive-slot, and
  ownership assertions to every stochastic trial.
- [x] Mark observations with `gpu_parity`, `stochastic`, and `cuda` as applicable;
  require Warp CPU when installed and skip CUDA only when unavailable.
- [x] Run focused deterministic/stochastic commands and the full coagulation
  suite without slow/performance markers.

### Documentation

- [x] Publish a matrix mapping each supported row to deterministic, stochastic,
  conservation, edge, and device evidence.
- [x] Document tolerances, seed/sample policy, Warp CPU requirement, optional
  CUDA behavior, known unsupported rows, and focused reproduction commands.
- [x] Validate documentation links, command names, pytest markers, and support
  wording for E5-F9 consumption.
