# Phase Details

- [ ] **E2-F5-P1:** Design scalar-to-environment compatibility contract with unit tests
  - Issue: TBD | Size: XS | Status: Not Started
  - Goal: Specify API behavior for scalar inputs, explicit environment inputs,
    conflict handling, and downstream kernel feed points.
  - Files: `particula/gpu/kernels/condensation.py`,
    `particula/gpu/kernels/coagulation.py`, relevant GPU kernel tests, and this
    plan's documentation sections as needed.
  - Tests: Add lightweight unit tests or parameterized expectations that lock
    scalar compatibility and explicit conflict/mismatch behavior where helpers
    already exist.

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
