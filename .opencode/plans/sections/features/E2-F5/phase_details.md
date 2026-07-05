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

- [x] **E2-F5-P2:** Implement environment normalization and validation helpers with unit tests
  - Issue: #1204 | Size: S | Status: Completed
  - Goal: Normalize scalar temperature/pressure to per-box arrays and validate
    `WarpEnvironmentData` or equivalent arrays against `n_boxes` and device.
  - Delivered: Added private module `particula/gpu/kernels/environment.py` with
    `_ensure_environment_arrays(...)`, `_validate_box_array(...)`, and
    `_is_warp_array_like(...)`; updated both GPU entry points to normalize once
    before downstream setup; enabled valid explicit `environment=...`, direct
    `(n_boxes,)` Warp-array inputs, and hybrid scalar-plus-array direct inputs.
    Condensation now prepares box properties once per call and coagulation now
    passes normalized arrays directly into `brownian_coagulation_kernel(...)`.
  - Files: `particula/gpu/kernels/environment.py`,
    `particula/gpu/kernels/condensation.py`,
    `particula/gpu/kernels/coagulation.py`,
    `particula/gpu/kernels/tests/environment_test.py`,
    `particula/gpu/kernels/tests/condensation_test.py`, and
    `particula/gpu/kernels/tests/coagulation_test.py`.
  - Tests: Added helper tests for scalar broadcast, valid direct arrays, valid
    `WarpEnvironmentData`, hybrid inputs, wrong shape, wrong `n_boxes`, wrong
    device, mixed direct-plus-environment ambiguity, and missing direct inputs;
    replaced P1 explicit-environment rejection tests with success coverage;
    added direct-array, hybrid-input, mismatch, and pre-launch short-circuit
    regressions for both entry points; added reuse/precompute regression checks.

- [ ] **E2-F5-P3:** Migrate condensation GPU API to per-box environment inputs with compatibility tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Remaining condensation-specific follow-up only if later work needs
    more than the shared helper path already shipped in issue #1204.
  - Files: `particula/gpu/kernels/condensation.py`,
    `particula/gpu/kernels/tests/condensation_test.py`, and exports only if a
    new wrapper API is introduced.
  - Tests: Existing scalar condensation tests still pass; uniform per-box
    environment matches scalar output; non-uniform multi-box environment path
    now executes; any future P3 work would be additive.

- [ ] **E2-F5-P4:** Migrate coagulation GPU API and document downstream environment handoff
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Remaining coagulation/documentation follow-up only if downstream
    development docs or new wrappers are needed beyond the shipped shared
    environment path.
  - Files: `particula/gpu/kernels/coagulation.py`,
    `particula/gpu/kernels/tests/coagulation_test.py`,
    `docs/Features/Roadmap/data-oriented-gpu.md` or the relevant E2 docs.
  - Tests: Existing scalar coagulation tests still pass; explicit environment,
    direct-array, hybrid-input, wrong-`n_boxes`, and device mismatch behavior
    are now covered at the kernel-test level.

## Phase Ordering Notes

- P1 should define the scalar-versus-explicit-environment contract before any
  helper code lands so later phases do not encode conflicting precedence rules.
- P2 should follow P1 and `E2-F3-P2` because environment normalization should
  reuse the published transfer-helper naming and device-validation surface.
- Issue #1204 pulled the minimum viable condensation/coagulation runtime
  migration into P2 so the shared helper could be exercised by real entry-point
  execution instead of a helper-only shell.
- Any later P3/P4 work should now stay narrow and additive rather than
  re-litigating the shared normalization contract.
