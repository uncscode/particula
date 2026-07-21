# Success Criteria

- [ ] E6-F1/T1 is an explicit upstream dependency and its finite-step equation,
  units, and validation semantics are the sole CPU parity oracle.
- [ ] A public low-level `dilution_step_gpu` accepts documented scalar and
  same-device per-box inputs and returns the same Warp container objects.
- [ ] Particle number concentration and scalar/multi-species gas mass
  concentration match the independent CPU reference at recorded float64
  tolerances for one and multiple boxes.
- [ ] Zero coefficient/flow-equivalent input and zero elapsed time are exact
  no-ops with no concentration or metadata writes.
- [ ] Particle mass, charge, density, distribution coordinates and volume remain
  unchanged; gas molar mass, vapor pressure, partitioning, names held by the CPU
  owner, and all shapes/devices/dtypes remain unchanged.
- [ ] Supplied Warp input arrays and all container fields retain identity; the
  step performs no hidden transfer, CPU fallback, resizing, or host conversion.
- [ ] Negative/nonfinite physical values, wrong types/shapes/dtypes/devices,
  inconsistent box dimensions, and invalid concentration state fail before any
  particle or gas mutation.
- [ ] Warp CPU parity is required and passes; CUDA executes the same matrix when
  available and skips cleanly otherwise.
- [ ] The direct API is exported through `particula.gpu.kernels`, focused tests
  pass with warnings as errors, changed code meets >=80% coverage, and no
  threshold is lowered.
- [ ] Documentation states explicit-transfer, support, and deferred boundaries
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
