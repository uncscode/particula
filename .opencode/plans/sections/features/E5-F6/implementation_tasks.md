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
- [ ] Reuse each sibling term's property preparation and majorant helper without
  recomputing shared radius, viscosity, environment, or active-index state.
- [ ] Add enabled component majorants into an fp64 `total_majorant` and guard
  non-finite, negative, overflowed, and zero-total cases.
- [ ] Add enabled component pair rates into one fp64 `total_pair_rate` for each
  candidate; retain a single ratio, acceptance draw, and swap-pop removal.
- [ ] Keep one `collision_pairs`, `n_collisions`, and `rng_states` path and one
  charge-aware apply launch. Do not add mechanism-specific output buffers.
- [ ] Update `coagulation_step_gpu` docstrings with matrix, required inputs,
  single-pass semantics, and compatibility behavior.

## Tooling / Tests

- [x] Add table-driven recognition, validation, and atomicity tests in
  `particula/gpu/kernels/tests/coagulation_test.py` for registered, rejected,
  and deferred masks.
- [ ] Build independent NumPy component-rate matrices and assert the device
  total equals their sum with explicit fp64 tolerances.
- [ ] Enumerate every active unordered pair in deterministic fixtures and prove
  `total_rate <= sum(component_majorants)` for two-way and four-way masks.
- [ ] Instrument test-only RNG/acceptance diagnostics to prove one candidate
  stream and one acceptance draw, without changing the public API.
- [ ] Add bounded stochastic tests over fixed seed sets for additive collision
  counts; use aggregate or sigma bounds rather than exact pair replay.
- [ ] Cover zero/one/two/many active particles, inactive gaps, mixed scales,
  one/multi-box inputs, capacity limits, and zero component contributions.
- [ ] Assert caller buffer identity, persistent RNG reuse/reset, sorted in-range
  disjoint accepted pairs, species-mass conservation, and charge conservation.
- [ ] Snapshot particle, output, and RNG buffers for every invalid additive call
  to prove fail-before-mutation behavior.
- [ ] Run focused Warp CPU tests, optional CUDA parameterization, the existing
  coagulation regression suite, and repository lint/type checks.
