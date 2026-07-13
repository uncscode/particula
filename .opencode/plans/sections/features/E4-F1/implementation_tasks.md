# Implementation Tasks

## GPU Physics and API

- [ ] Choose and document stable integer codes for constant and Buck models.
- [ ] Add a fixed-shape thermodynamic configuration with typed species-indexed
  mode and parameter arrays.
- [ ] Add validation for required configuration, supported modes, finite and
  nonnegative model parameters, dtype, shape, species count, ordering contract,
  and active Warp device.
- [ ] Implement constant and piecewise Buck Warp calculations from the exact CPU
  references using `float64`.
- [ ] Implement a refresh launch that writes every `(box, species)` output slot.
- [ ] Add keyword-only configuration to `condensation_step_gpu()` without
  breaking current positional calls.
- [ ] Invoke refresh after `_ensure_environment_arrays()` and before the current
  mass-transfer kernel.
- [ ] Ensure all validation failures occur before `gas.vapor_pressure`, gas
  concentration, or particle mass can mutate.

## Tooling / Tests

- [ ] Add formula unit tests to `particula/gpu/kernels/tests/condensation_test.py`
  or a focused `*_test.py` beside the new module.
- [ ] Add mixed-model, multi-box, freezing-boundary, repeated-temperature, and
  species-order tests.
- [ ] Add invalid/missing configuration and failure-before-mutation tests.
- [ ] Run focused Warp CPU tests and optional CUDA parity with explicit tolerances.
- [ ] Run Ruff, mypy for changed source, and the repository's fast test suite.

## Documentation

- [ ] Document mode codes, parameter units/shapes, derived-buffer ownership, and
  refresh timing.
- [ ] Cross-reference E4-F2 through E4-F7 deferred responsibilities.
