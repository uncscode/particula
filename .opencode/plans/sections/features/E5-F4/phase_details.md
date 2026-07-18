# Phase Details

- [x] **E5-F4-P1:** Port mixture-density, settling-velocity, and SP2016 pair physics with unit tests
  - Issue: #1347 | Size: S | Status: Complete
  - Goal: Add scalar fp64 Warp helpers for effective particle density, Stokes
    settling with Cunningham slip correction, and
    `pi * (r_i + r_j)^2 * abs(v_i - v_j)` with efficiency fixed at 1.
  - Files: `particula/gpu/dynamics/coagulation_funcs.py`,
    `particula/gpu/dynamics/tests/coagulation_funcs_test.py`
  - Delivered: internal device-only helpers and direct Warp probes; no public
    dispatch, mechanism registration, majorant, or runnable API.
  - Tests: independent NumPy fixtures for single- and multi-species density,
    radius/settling properties, symmetry, zero equal-velocity rate, batched
    invalid/overflow/underflow safe-zero behavior, Warp CPU, and optional CUDA.

- [x] **E5-F4-P2:** Add a safe sedimentation majorant and bounded internal dispatch with unit tests
  - Issue: #1348 | Size: S | Status: Complete
  - Goal: Compute the maximum sedimentation rate across all active unordered
    pairs and route the term through E5-F1's shared rate/majorant dispatcher
    without adding a second candidate or RNG pass.
  - Files: `particula/gpu/kernels/coagulation.py`,
    `particula/gpu/kernels/tests/coagulation_test.py`
  - Delivered: exact sedimentation-only private dispatch through the shared
    scheduler/RNG path, exhaustive compact active-pair majorant, and private
    cleared settling-velocity scratch. Public preflight still rejects
    sedimentation; mixed sedimentation masks are no-ops.
  - Tests: co-located majorant, scheduler/RNG, scratch, and mixed-mask
    state-preservation regressions.

- [x] **E5-F4-P3:** Integrate sedimentation execution with multi-box conservation and state-safety tests
  - Issue: #1349 | Size: S | Status: Complete
  - Delivered: Registered public particle-resolved
    `("sedimentation_sp2016",)` execution, sedimentation-specific domain
    preflight, and atomic rejected-call behavior before output/RNG/mutation.
  - Files: `particula/gpu/kernels/coagulation.py`,
    `particula/gpu/kernels/tests/coagulation_test.py`
  - Tests: independent pair parity, stochastic aggregate evidence, multi-box
    direct/environment cases, per-box/species conservation, donor clearing,
    inactive slots, caller-buffer identity, persistent RNG reuse/reset, and
    unchanged particle/output/RNG snapshots for invalid or unsupported calls.

- [x] **E5-F4-P4:** Update development documentation
  - Issue: #1349 | Size: XS | Status: Complete
  - Delivered: Published the canonical direct-kernel configuration and ownership,
    fp64/unit-efficiency, no-transfer/no-fallback, and deferred-mode limits.
  - Files: `docs/Features/data-containers-and-gpu-foundations.md`,
    `docs/Features/Roadmap/data-oriented-gpu.md`, relevant API docstrings, and
    E5/E5-F4 plan sections.
   - Evidence: canonical support wording is recorded in the user-facing GPU
     direct-kernel documentation.

- [x] **Issue #1350 follow-up:** Complete sedimentation documentation detail
  - Status: Complete | Date: 2026-07-17
  - Delivered: Added the SP2016 pair equation, m³/s units, Seinfeld & Pandis
    citation, fixed-efficiency/no-argument scope, supported-float direct
    thermodynamic-input boundary, private fp64 scratch distinction, and focused
    sedimentation test evidence wording.
  - Confirmed: Caller-owned output/RNG identity, same-device environment,
    sedimentation preflight ordering, Warp CPU baseline, and optional
    cleanly-skipped CUDA evidence. This follow-up supplements rather than
    replaces #1349's completed P3/P4 delivery.
