# Implementation Tasks

## GPU Backend

- [x] Define canonical mechanism names, bit flags, ordering, and the frozen
  mechanism configuration in `particula/gpu/kernels/coagulation.py`.
- [x] Implement a pure host validator/resolver for non-empty, known, unique
  mechanisms and `distribution_type="particle_resolved"`.
- [x] Implement an explicit executable-capability table that initially enables
  only Brownian and reports the owning E5 track for reserved terms.
- [x] Add keyword-only configuration to `coagulation_step_gpu`; resolve and
  validate it before particle/environment normalization, allocations, RNG
  initialization, or launch.
- [x] Add private mask-dispatch helpers in
  `particula/gpu/kernels/coagulation.py` for sanitized finite-positive Brownian
  pair rates and safe majorants; keep Brownian physics in
  `particula/gpu/dynamics/coagulation_funcs.py`.
- [x] Refactor `brownian_coagulation_kernel` to consume one total pair rate and
  one total majorant without changing active-pair selection or swap-pop rules.
- [ ] Keep `apply_coagulation_kernel` behavior unchanged in E5-F1; charge merge
  semantics belong to E5-F2.
- [x] Guard zero, non-finite, negative, and underestimated rate/majorant paths
  before acceptance, collision output, counters, or swap-pop mutation.

## Tooling / Tests

- [x] Add resolver table tests to
  `particula/gpu/kernels/tests/coagulation_test.py` for defaults, canonical
  order, duplicate/empty/unknown terms, reserved terms, and unsupported modes.
- [x] Add test-only synthetic additive probes proving two terms are summed
  before one acceptance comparison rather than sampled separately.
- [x] Compare omitted configuration and explicit Brownian calls under equal
  seeds, buffers, scalar environment inputs, and per-box environment arrays.
- [x] Snapshot masses, concentration, charge, collision pairs/counts, and
  persistent RNG before every public preflight failure and assert identity and
  values remain unchanged.
- [x] Add focused unit, fixed-seed deterministic, and marked stochastic Warp
  coverage in `particula/gpu/kernels/tests/coagulation_test.py`.
- [ ] Run focused tests, Ruff, and mypy without changing coverage thresholds.

## Documentation

- [x] Document the extension checklist: identifier, required inputs, property
  preparation, pair term, safe majorant, capability row, and co-located tests.
- [x] Publish the initial supported/reserved matrix and explicitly state that
  binned and continuous-PDF GPU coagulation are rejected.
- [x] Record additive single-pass semantics for E5-F3 through E5-F7 and update
  plan phase statuses as work ships.
