# Implementation Tasks

## GPU Backend

- [x] Add a private immutable P1 recognition table in
  `particula/gpu/kernels/coagulation.py` for singleton, pair, and four-term
  masks, distinct from the executable capability gate; reject three-term masks
  before particle access.
- [x] Centralize enabled-bit read-only preflight so turbulent, charged, and
  sedimentation requirements are checked before volume/environment
  normalization, output/RNG work, allocations, or selection launch. Valid
  recognized deferred masks raise the stable deferred-execution error.
- [x] Reuse each sibling term's property preparation and majorant helper without
  recomputing shared radius, viscosity, environment, or active-index state.
- [x] Add enabled component majorants into a private fp64 `total_majorant`
  dispatch for all recognized two-way/four-way masks. Checked addition fails
  closed for non-finite, nonpositive, and overflowed aggregates.
- [x] Add enabled component pair rates into one private fp64 `total_pair_rate`
  dispatch for each candidate. The acceptance guard admits finite positive
  ratios only, permits exactly the eight-ULP roundoff allowance, and rejects
  before any draw/write/removal on a material bound violation.
- [x] Keep one `collision_pairs`, `n_collisions`, and `rng_states` path and one
  charge-aware apply launch. Do not add mechanism-specific output buffers.
- [x] Update `coagulation_step_gpu` docstrings with matrix, required inputs,
  single-pass semantics, and compatibility behavior.

## Tooling / Tests

- [x] Add table-driven recognition, validation, and atomicity tests in
  `particula/gpu/kernels/tests/coagulation_test.py` for registered, rejected,
  and deferred masks.
- [x] Build independent deterministic NumPy component-rate oracles and assert
  private Warp totals equal their sums with explicit fp64 tolerances.
- [x] Enumerate active unordered pairs in sparse, mixed-scale fixtures and prove
  `total_rate <= sum(component_majorants)` for every recognized two-way and
  four-way mask.
- [x] Add private selector/acceptance diagnostics and regressions proving a
  valid proposal has one candidate stream and at most one acceptance draw,
  while invalid/materially unbounded proposals cannot draw or mutate state.
- [x] Add bounded stochastic tests over fixed seed sets for additive collision
  counts; use aggregate or sigma bounds rather than exact pair replay.
- [x] Cover zero/one/two/many active particles, inactive gaps, mixed scales,
  one/multi-box inputs, capacity limits, and zero component contributions.
- [x] Assert caller buffer identity, persistent RNG reuse/reset, sorted in-range
  disjoint accepted pairs, species-mass conservation, and charge conservation.
- [x] Snapshot particle, output, and RNG buffers for every invalid additive call
  to prove fail-before-mutation behavior.
- [x] Run focused Warp CPU tests, optional CUDA parameterization, the existing
  coagulation regression suite, and repository lint/type checks.
