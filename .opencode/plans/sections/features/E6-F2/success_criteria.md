# Success Criteria

## P1 Status (#1395)

- [x] Concrete-module-only `dilution_step_gpu(particles, gas, coefficient,
  time_step)` accepts finite nonnegative real scalar metadata or a same-device
  `wp.float64` `(n_boxes,)` coefficient array, and returns identical containers
  without a launch or caller-state write.
- [x] P1 tests cover direct import/no package re-export, scalar normalization,
  per-box identity, zero/no-write paths, and scalar/metadata rejection order.
- [x] P1 explicitly carries `alpha = Q / V` [s^-1] and the future P2 update
  `c_new = c * exp(-alpha * time_step)` without executing it.

## P3 Status (#1397)

- [x] Entry-point preflight validates coefficient form, time, masses, per-box
  coefficient schema/values, particle concentration schema/values, then gas
  concentration schema/values before an allocation, kernel launch, or write.
- [x] Masses, coefficients, and concentration fields use their exact
  same-device `wp.float64` Warp schemas; physical coefficient and concentration
  values are finite and nonnegative.
- [x] Rejections and valid scalar-zero/zero-time no-ops complete full
  validation without private allocation or launch, as covered by state snapshots
  and allocation/launch spies. Post-launch rollback remains deferred.

## Feature Completion Criteria

- [x] E6-F1/T1 is an explicit upstream dependency and its finite-step equation,
  units, and validation semantics are the sole CPU parity oracle.
- [x] A public low-level `dilution_step_gpu` accepts documented scalar and
  same-device per-box inputs and returns the same Warp container objects.
- [x] Particle number concentration and scalar/multi-species gas mass
  concentration match the independent NumPy reference at `rtol=1e-12`,
  `atol=0` for one and multiple boxes (P4, #1398).
- [x] Zero coefficient/flow-equivalent input and zero elapsed time are exact
  no-ops with no concentration or metadata writes (P4, #1398).
- [x] Particle mass, charge, density, distribution coordinates and volume remain
  unchanged; gas molar mass, vapor pressure, partitioning, names held by the CPU
  owner, and all shapes/devices/dtypes remain unchanged (P4 invariants, #1398).
- [x] Supplied Warp input arrays and all container fields retain identity; the
  step performs no hidden transfer, CPU fallback, resizing, or host conversion.
- [x] Negative/nonfinite physical values, wrong types/shapes/dtypes/devices,
  inconsistent box dimensions, and invalid concentration state fail before any
  particle or gas mutation.
- [x] Warp CPU parity is required and passes; CUDA executes the same matrix when
  available and skips cleanly otherwise (P4, #1398).
- [x] The direct API is exported through `particula.gpu.kernels`, focused tests
  pass with warnings as errors, changed code meets >=80% coverage, and no
  threshold is lowered.
- [x] Documentation states explicit-transfer, support, and deferred boundaries
  and does not claim an Epic G runnable, scheduler, or performance capability.

## Metrics

| Metric | Baseline | Target | Source |
|---|---|---|---|
| Direct GPU dilution APIs | 0 | 1 supported low-level step | Import/API tests |
| CPU/Warp deterministic acceptance cases | 0 | 100% passing | Focused parity matrix |
| Exact zero-input no-op failures | Unspecified | 0 | Snapshot/no-launch tests |
| Protected-field or identity violations | Unspecified | 0 | Invariant tests |
| Invalid-call partial mutations | Unspecified | 0 | Atomicity tests |
| Hidden process-level transfers/fallbacks | N/A | 0 | Boundary tests/review |
| Required Warp CPU environments passing | 0 | 100% | CI pytest results |
| Changed-code coverage | N/A | >=80%, no reduction | pytest-cov/CI |
