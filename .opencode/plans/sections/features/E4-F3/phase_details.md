# Phase Details

- [ ] **E4-F3-P1:** Reusable stable-shape scratch buffers and pre-mutation validation with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Add optional caller-owned scratch inputs and validate shape, dtype,
    and device before any buffer or particle mutation.
  - Files: `particula/gpu/kernels/condensation.py`,
    `particula/gpu/kernels/tests/condensation_test.py`
  - Tests: identity reuse, wrong shape/device/dtype, unchanged buffers and
    particles on rejection, and allocation-free supplied-scratch path.

- [ ] **E4-F3-P2:** Fixed four-substep integration and per-substep physics refresh with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Run four equal calculate/apply iterations, refresh E4-F1 physics each
    iteration, and return accumulated total transfer.
  - Files: `particula/gpu/kernels/condensation.py`, production contract tests
  - Tests: exactly four launches, equal durations, current-state refresh,
    accumulation semantics, clamp behavior, determinism, and finite output.

- [ ] **E4-F3-P3:** Promote issue 1272 stiffness and buffer-reuse validation coverage
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Move selected candidate evidence onto the production Warp path without
    weakening recorded validation signals.
  - Files: `particula/gpu/kernels/tests/_condensation_test_support.py`,
    `particula/gpu/kernels/tests/condensation_stiffness_test.py`
  - Tests: nanometer, accumulation-mode, and droplet-like grids; `rtol=5e-2`
    recorded bounds; Warp CPU parity; optional CUDA; stable repeated reuse.

- [ ] **E4-F3-P4:** Update development documentation for fixed-four integration
  - Issue: TBD | Size: XS | Status: Not Started
  - Goal: Document selected production behavior, scratch ownership, limitations,
    and focused validation commands.
  - Files: `docs/Features/Roadmap/condensation-stiffness-study.md`,
    `docs/Features/Roadmap/data-oriented-gpu.md`, relevant feature docs
  - Tests: Markdown links and documented command/reference verification.
