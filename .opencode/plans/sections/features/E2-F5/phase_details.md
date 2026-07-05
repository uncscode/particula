# Phase Details

- [x] **E2-F5-P1:** Design scalar-to-environment compatibility contract with unit tests
  - Issue: #1203 | Size: XS | Status: Completed
  - Goal: Specify API behavior for scalar inputs, explicit environment inputs,
    conflict handling, and downstream kernel feed points.
  - Delivered: Added keyword-only `environment` parameters and phase-scoped
    docstrings to `condensation_step_gpu(...)` and `coagulation_step_gpu(...)`;
    added early mixed-input and pure explicit-environment guards before helper
    calls or Warp launch setup; left real per-box environment execution for
    later phases.
  - Files: `particula/gpu/kernels/condensation.py`,
    `particula/gpu/kernels/coagulation.py`,
    `particula/gpu/kernels/tests/condensation_test.py`, and
    `particula/gpu/kernels/tests/coagulation_test.py`.
  - Tests: Added keyword-only signature checks, legacy positional scalar-call
    regression checks, parametrized mixed-input `ValueError` coverage, pure
    explicit-environment P1 rejection checks, and short-circuit tests proving
    contract errors fire before gas-property helpers, volume normalization, or
    Warp launch.

- [ ] **E2-F5-P2:** Implement environment normalization and validation helpers with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Normalize scalar temperature/pressure to per-box arrays and validate
    `WarpEnvironmentData` or equivalent arrays against `n_boxes` and device.
  - Files: `particula/gpu/kernels/condensation.py`,
    `particula/gpu/kernels/coagulation.py`, `particula/gpu/warp_types.py`,
    `particula/gpu/conversion.py` if E2-F2 exposes conversion requirements.
  - Tests: Helper tests for scalar broadcast, `(n_boxes,)` acceptance,
    wrong-shape `ValueError`, wrong-device `ValueError`, and optional explicit
    environment precedence/conflict behavior.

- [ ] **E2-F5-P3:** Migrate condensation GPU API to per-box environment inputs with compatibility tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Update condensation launch/kernel inputs so box-local environment
    values are available while scalar callers remain supported.
  - Files: `particula/gpu/kernels/condensation.py`,
    `particula/gpu/kernels/tests/condensation_test.py`, and exports only if a
    new wrapper API is introduced.
  - Tests: Existing scalar condensation tests still pass; uniform per-box
    environment matches scalar output; non-uniform multi-box environment path
    executes; environment `n_boxes` mismatch raises before launch.

- [ ] **E2-F5-P4:** Migrate coagulation GPU API and document downstream environment handoff
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Update coagulation launch/kernel inputs to consume per-box environment
    values and update development docs so later physics kernels use the same
    path.
  - Files: `particula/gpu/kernels/coagulation.py`,
    `particula/gpu/kernels/tests/coagulation_test.py`,
    `docs/Features/Roadmap/data-oriented-gpu.md` or the relevant E2 docs.
  - Tests: Existing scalar coagulation tests still pass; uniform per-box
    environment matches scalar behavior within stochastic tolerances; wrong
    `n_boxes` and device mismatch errors are covered.

## Phase Ordering Notes

- P1 should define the scalar-versus-explicit-environment contract before any
  helper code lands so later phases do not encode conflicting precedence rules.
- P2 should follow P1 and `E2-F3-P2` because environment normalization should
  reuse the published transfer-helper naming and device-validation surface.
- P3 should follow P2 and the `E2-F4-P3` vapor-pressure decision so the
  condensation API does not freeze an environment contract that sibling gas work
  still treats as provisional.
- P4 should follow P3 so coagulation inherits the tested normalization path and
  documentation publishes one shared per-box environment handoff.
