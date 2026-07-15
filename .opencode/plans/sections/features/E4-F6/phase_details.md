# Phase Details

Phase issue creation is intentionally deferred until E4 implementation issues
are generated and scheduled; `TBD` is not an unresolved design decision.

- [x] **E4-F6-P1:** Add device-aware condensation parity matrix with independent CPU references
  - Issue: #1308 | Size: S | Status: Completed
  - Delivered: Two shared deterministic fp64 cases compare the direct GPU step
    with an independent NumPy fixed-four-substep/P2/gas-coupled oracle on Warp
    CPU and, when available, CUDA.
  - Files: `particula/gpu/kernels/tests/_condensation_test_support.py`, `condensation_test.py`
  - Tests: One-box uptake/latent-heat/inventory-limited and multi-box,
    multi-species mixed-phase/gated cases; mass and gas are asserted separately
    at `rtol=1e-10` with per-output scale-derived finite `atol`. Coverage
    includes uptake, evaporation, disabled partitioning, zero gas, and inactive
    particle slots; CUDA skips cleanly when unavailable.

- [x] **E4-F6-P2:** Add per-box per-species conservation and mutation-contract regressions
  - Issue: #1309 | Size: S | Status: Completed
  - Delivered: Warp-CPU contract regressions separately prove
    concentration-weighted particle-plus-gas conservation for each box/species,
    P2-finalized total-transfer accounting, and unweighted latent-energy
    accounting.
  - Files: `particula/gpu/kernels/tests/_condensation_test_support.py`,
    `particula/gpu/kernels/tests/condensation_test.py`
  - Tests: Inactive, disabled, and zero-concentration entries; inventory-limited
    uptake; finite/nonnegative final state; immutable caller inputs;
    caller-owned output identity; atomic representative invalid-buffer paths;
    and deterministic runs with fresh state and sidecars. No production API or
    physics changes were made.

- [x] **E4-F6-P3:** Record fixed-loop graph-capture capability boundary
  - Issue: #1310 | Size: S | Status: Completed
  - Delivered: `condensation_graph_capture_test.py` establishes the public
    step's capture limitation, rather than capture readiness.
  - Files: `particula/gpu/kernels/tests/condensation_graph_capture_test.py`
  - Tests: Warp CPU is capability-skipped because Warp capture requires CUDA.
    CUDA public-step replay is strict-xfailed because host validation readbacks
    are not capture-safe. These guarded outcomes document unsupported capture,
    not successful replay. No production API or behavior changed.

- [x] **E4-F6-P4:** Add bounded autodiff-readiness experiments and limitation tests
  - Issue: #1311 | Size: S | Status: Completed
  - Delivered: `condensation_autodiff_test.py` records the derivative of the
    out-of-place raw `condensation_mass_transfer_kernel` proposal with respect
    to gas concentration. A one-box fp64 Warp Tape result is compared with a
    centered finite difference while executable positivity/inventory margins
    keep P2 boundaries inactive.
  - Files: `particula/gpu/kernels/tests/condensation_autodiff_test.py`
  - Tests: Warp CPU and optional CUDA Tape/reference checks; scoped
    `verify_autograd_array_access` restoration after a sentinel failure; and
    forward-only P2 evaporation clamp, uptake inventory scaling, and in-place
    mutation tests. Missing Tape, backward, or access-verification capability
    skips with bounded-probe/device context. No production API, behavior, or
    published documentation changed.

- [x] **E4-F6-P5:** Update development documentation and evidence matrix
  - Issue: #1312 | Size: XS | Status: Completed
  - Delivered: Published the P1--P4 matrix, supported-backend convention, and
    focused commands while documenting P3 CPU capability skip/CUDA strict-xfail
    as unsupported public-step capture, not replay support.
  - Files: `docs/Features/condensation_strategy_system.md`,
    `docs/Features/Roadmap/warp-autodiff-limitations.md`,
    `docs/Features/Roadmap/condensation-stiffness-study.md`, and
    `.opencode/guides/testing_guide.md`.
  - Validation: Documentation points to discoverable `*_test.py` wrappers and
    retains P1/P2/P3/P4 evidence and non-claim boundaries.
