# Implementation Tasks

## E2-F5-P1: Compatibility Contract

- [x] Confirm E2-F2's environment container names and fields.
- [x] Reserve explicit environment input through an optional keyword-only
  `environment` argument on both GPU entry points instead of a new wrapper API.
- [x] Define conflict handling for scalar values plus explicit environment:
  any mixed call raises a stable early `ValueError`.
- [x] Define the temporary pure explicit-environment behavior for P1:
  `temperature=None`, `pressure=None`, and `environment=...` raises a
  phase-scoped early `ValueError` until later migrations wire in per-box
  execution.
- [x] Add or update tests that assert legacy scalar calls still use the old
  public call shape.
- [x] Add short-circuit regression tests proving invalid contract calls fail
  before host-side helper execution or Warp launch setup.

## E2-F5-P2: Helpers and Validation

- [x] Add `_ensure_environment_arrays(...)` plus supporting private helpers in
  `particula/gpu/kernels/environment.py`.
- [x] Validate temperature and pressure arrays as `(n_boxes,)`.
- [x] Validate device consistency with particle and gas Warp arrays.
- [x] Add unit tests for scalar broadcast, valid arrays, wrong shape, wrong
  device, missing-input, and ambiguous-input behavior.

## E2-F5-P3: Condensation Migration

- [x] Route `condensation_step_gpu` through the normalization helper.
- [x] Update kernel/runtime inputs to receive per-box temperature and pressure
  through precomputed box-property arrays.
- [x] Ensure scalar temperature/pressure tests in condensation continue passing.
- [x] Add uniform-environment equivalence and non-uniform multi-box execution
  tests.
- [x] Add pre-launch mismatch tests for `environment.n_boxes != particles.n_boxes`.

## E2-F5-P4: Coagulation Migration and Handoff

- [x] Route `coagulation_step_gpu` through the same environment normalization
  path.
- [x] Update Brownian coagulation kernel logic to use `box_idx` environment
  values.
- [x] Preserve stochastic scalar tests and add explicit-environment/direct-array
  compatibility coverage.
- [ ] Document downstream expectations for later GPU physics kernels.
- [x] Ensure code-local docstrings describe compatibility behavior clearly.
