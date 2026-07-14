# Phase Details

Phase issue creation is intentionally deferred until E4 implementation issues
are generated and scheduled; `TBD` is not an unresolved design decision.

- [x] **E4-F3-P1:** Reusable stable-shape scratch buffers and pre-mutation validation with unit tests
  - Issue: #1292 | Size: S | Status: Shipped (2026-07-13)
  - Delivered: Added the concrete-module-only frozen
    `CondensationScratchBuffers` sidecar and keyword-only `scratch_buffers` to
    the existing one-update entry point. All supplied fields validate atomically
    for shape, fp64 dtype, and active device before any fallback allocation,
    normalization, refresh, launch, clear, or mutation.
  - Files: `particula/gpu/kernels/condensation.py`,
    `particula/gpu/kernels/tests/condensation_test.py`
  - Tests: cover complete and partial sidecars, all supported environment input
    forms, identity and stable shapes, transfer overlap rejection, wrong
    type/shape/dtype/device, unchanged state after rejection, and the
    allocation-free complete-sidecar path.

- [x] **E4-F3-P2:** Fixed four-substep integration and per-substep physics refresh with unit tests
  - Issue: #1293 | Size: S | Status: Shipped (2026-07-13)
  - Delivered: `condensation_step_gpu()` unconditionally schedules four
    `time_step / 4.0` iterations. Each iteration refreshes thermodynamics and
    environment properties, calculates from predecessor-updated mass, clamps
    applied transfer to available mass, and accumulates it in total storage.
    Work storage retains the final raw proposal.
  - Files: `particula/gpu/kernels/condensation.py`,
    `particula/gpu/kernels/tests/condensation_test.py`,
    `particula/gpu/kernels/tests/_condensation_test_support.py`
  - Tests: fixed integration launch/order behavior, current-state refresh,
    applied-total and raw-work semantics, forced evaporation clamp behavior,
    determinism, finite nonnegative mass, and unchanged gas concentration.

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
