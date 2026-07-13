# Testing Strategy

Every implementation phase ships with its tests; coverage thresholds remain
unchanged and changed code must retain at least 80% coverage.

## Per-Phase Approach

- **P1:** In `_condensation_test_support.py` plus `condensation_test.py`, compare
  deterministic fp64 production outputs to independent one-box CPU equations.
  Run one/multi-box and multi-species cases on Warp CPU and optional CUDA with
  parity `rtol=1e-10` and a scale-derived `atol`.
- **P2:** Add separate strict per-box/per-species inventory and latent-energy
  assertions with `rtol=1e-12` and
  `atol=max(1e-18, scale * eps)`. Cover inactive entries, clamp bounds,
  deterministic replay, immutable inputs, and fail-before-mutation paths.
- **P3:** Add
  `particula/gpu/kernels/tests/condensation_graph_capture_test.py`. Preallocate
  all scratch, capture exactly four substeps, replay repeatedly, and compare
  with a normal launch for state, transfer, conservation, identity, and shape.
- **P4:** Add
  `particula/gpu/kernels/tests/condensation_autodiff_test.py` for a bounded
  out-of-place smooth-interior tape/gradcheck and access verification.
  Unsupported backends skip with precise reasons; clamps, inventory gates, and
  in-place mutation are documented as unsupported rather than assigned
  permissive gradient tolerances.
- **P5:** Validate documentation links and execute focused reproduction commands.

Parity and conservation are always distinct assertions. Warp absence and CUDA
absence skip cleanly according to repository policy, but when Warp is installed
the CPU backend is mandatory. No benchmark or stochastic marker is required.
