# Implementation Tasks

## GPU Backend

- [ ] Extend E5-F1's capability table in
  `particula/gpu/kernels/coagulation.py` with named approved two-way masks and
  the full four-way mask; preserve canonical ordering and single-term rows.
- [ ] Centralize per-mask required-input validation so every enabled mechanism
  is checked before volume normalization, RNG initialization, allocations, or
  selection launch.
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

- [ ] Add table-driven configuration tests in
  `particula/gpu/kernels/tests/coagulation_test.py` for every registered and
  deliberately unsupported mask.
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
