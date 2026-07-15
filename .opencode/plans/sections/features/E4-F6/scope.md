# Scope

This feature qualifies the completed E4 condensation path across devices and
execution modes; it primarily adds test infrastructure, regressions, and
evidence documentation rather than new physics.

## In Scope

- Deterministic fp64 one-box and multi-box parity against independent CPU
  equations for the combined E4-F1 through E4-F5 behavior.
- Mandatory Warp CPU and cleanly optional CUDA parameterization.
- **Delivered in P1 (#1308):** two shared one-/multi-box, multi-species cases
  in `particula/gpu/kernels/tests/_condensation_test_support.py`, re-exported
  by `condensation_test.py`; they compare particle mass and gas concentration
  independently to the NumPy four-substep/P2/gas-coupled oracle.
- Separate, strict per-box/per-species particle-plus-gas conservation and
  latent-energy bookkeeping assertions.
- Validation-before-mutation, unchanged-input, deterministic-repeat, and
  caller-owned stable scratch-buffer checks.
- **Delivered in P2 (#1309):** Warp-CPU public-step regressions in
  `particula/gpu/kernels/tests/_condensation_test_support.py`, exported by
  `condensation_test.py`, cover those accounting and mutation contracts with
  no production source or public API changes.
- Capture/replay evidence for the exactly-four-substep path where supported.
- **Delivered in P3 (#1310):** test-only capture/replay coverage in
  `particula/gpu/kernels/tests/condensation_graph_capture_test.py` uses all
  seven supplied scratch fields plus energy sidecars, two device-reset replays,
  and normal/replay state, invariant, and buffer-contract checks. It does not
  claim allocation-free capture or alter production behavior.
- Bounded tape/gradcheck or access-verification experiments, with clamps and
  in-place mutation limitations recorded explicitly.
- An evidence matrix and focused reproduction commands.

## Out of Scope

- New condensation physics, adaptive stepping, or data-dependent loop counts.
- Changes to CPU/GPU container schemas or hidden host/device synchronization.
- Requiring CUDA in CI, performance targets, or new diagnostics APIs.
- Claiming end-to-end production differentiability where Warp or clamp
  semantics do not support it.
- Work assigned to downstream E4-F7.
